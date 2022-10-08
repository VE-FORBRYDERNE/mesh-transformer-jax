import ray
import time
import numpy as np
from queue import Queue
from typing import Optional
import functools

from mesh_transformer.util import head_print, process_index


@ray.remote(resources={"TPU": 1})
class NetworkRunner(object):
    def __init__(self, mesh_shape, network_builder):
        self.mesh_shape = mesh_shape
        self.network_builder = network_builder

        self.input_q = Queue(maxsize=1)
        self.output_q = Queue(maxsize=1)

    def run(self):
        global print
        if not getattr(print, "_train_actor_patched", False):
            old_print = print
            @functools.wraps(print)
            def new_print(*args, **kwargs):
                old_print(f"[NODE {process_index()}]", *args, **kwargs)
            new_print._train_actor_patched = True
            print = new_print

        print(f"jax runtime initialization starting on node {process_index()}")
        import jax
        from jax.experimental.maps import thread_resources, ResourceEnv, Mesh
        import haiku as hk
        # jax.experimental.maps.EXPERIMENTAL_SPMD_LOWERING = True

        # thread_resources.env = ResourceEnv(Mesh(np.empty((), dtype=object), ()), ())
        thread_resources.env = ResourceEnv(Mesh(np.empty((), dtype=object), ()))

        start = time.time()
        jax.devices()

        import warnings
        # warnings.filterwarnings("ignore")
        warnings.filterwarnings("ignore", category=ResourceWarning)

        # if jax.host_id() == 0:
        #     warnings.filterwarnings("default")

        head_print(f"jax devices: {jax.device_count()} {jax.devices()}")
        head_print(f"jax runtime initialized in {time.time() - start:.06}s")
        devices = np.array(jax.devices()).reshape(self.mesh_shape)

        with jax.experimental.maps.mesh(devices, ('dp', 'mp')):
            start = time.time()
            network = self.network_builder()
            head_print(f"Initialized in {time.time() - start:.06}s")

            while True:
                operation, input = self.input_q.get()
                if operation == "train":
                    self.output_q.put(network.train(*input))
                elif operation == "eval":
                    self.output_q.put(network.eval(input))
                elif operation == "generate":
                    self.output_q.put(network.generate(*input))
                elif operation == "write_ckpt":
                    path, shard, save_opt_state = input
                    network.write_ckpt(path, shard, save_opt_state)
                    self.output_q.put(None)
                elif operation == "load_ckpt":
                    input, finetuning = input
                    network.load_ckpt(input, finetuning)
                    self.output_q.put((network.state["step"][0], network.state["opt_state"][-1].count[0]))
                elif operation == "get_params":
                    self.output_q.put(hk.data_structures.tree_size(network.state['params']))
                elif operation == "move_params":
                    # only needed for inference, otherwise first train step does this
                    local_shards = max(jax.local_device_count() // self.mesh_shape[1], 1)

                    # delete the optimizer states otherwise it OOMs for some reason
                    # TODO: use ShardedDeviceArray or something to get around this for bigger models
                    del network.state["opt_state"]
                    network.state = network.move_xmap(network.state, np.zeros(local_shards))
                    self.output_q.put(None)
                else:
                    raise Exception("Not implemented")

    def get_params(self):
        self.input_q.put(("get_params", None))
        return self.output_q.get()

    def train(self, sample, chunks):
        self.input_q.put(("train", (sample, chunks)))
        return self.output_q.get()

    def eval(self, sample):
        self.input_q.put(("eval", sample))
        return self.output_q.get()

    def generate(self, input):
        self.input_q.put(("generate", input))
        return self.output_q.get()

    def write_ckpt(self, path, shard, save_opt_state):
        self.input_q.put(("write_ckpt", (path, shard, save_opt_state)))
        return self.output_q.get()

    def load_ckpt(self, path, finetuning):
        self.input_q.put(("load_ckpt", (path, finetuning)))
        return self.output_q.get()

    def move_params(self):
        self.input_q.put(("move_params", None))
        return self.output_q.get()

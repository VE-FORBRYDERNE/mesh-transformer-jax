import functools
import multiprocessing
import time

import optax
import ray

from mesh_transformer import util
from mesh_transformer.TPU_cluster import TPUCluster
from mesh_transformer.transformer_shard import CausalTransformer, CausalTransformerV2
from mesh_transformer.util import clip_by_global_norm, additive_weight_decay
from ray_tpu import create_tpu, wait_til, get_connection, start_ray
from mesh_transformer.sampling import nucleaus_sample


def build_model(params, tpu_name, region, preemptible, version=1):
    gradient_accumulation_steps = params.get("gradient_accumulation_steps", 1)
    cores_per_replica = params["cores_per_replica"]
    tpu_size = params["tpu_size"]

    warmup_steps = params["warmup_steps"]
    anneal_steps = params["anneal_steps"]
    lr = params["lr"]
    end_lr = params["end_lr"]
    weight_decay = params["weight_decay"]

    assert tpu_size in [8, 32, 128, 256, 512]

    # create_tpu(tpu_name, region, f"v3-{tpu_size}", preemptible)
    print('—— SKIPPING TPU POD CREATION')
    assert wait_til(tpu_name, region, {'state': 'READY', 'health': 'HEALTHY'})

    conns = get_connection(tpu_name, region)

    print('———— GOT CONNECTIONS:',len(conns))

    assert len(conns) * 8 == tpu_size, "wrong size TPU for config"

    head_info = ray.init(
        include_dashboard=False,
        address="auto"
    )

    print('———— HEAD INFO:', head_info)

    address = head_info['address']
    print('head_info[address]:', address)
    print('—— BUILDING MODEL len conns', len(conns))

    time.sleep(3)
    # return

    print('WITH MULTI')
    with multiprocessing.pool.ThreadPool(processes=len(conns)) as p:
        p.map(functools.partial(start_ray, address=address, version=version), conns)
    
    scheduler = util.gpt3_schedule(warmup_steps, anneal_steps, lr, end_lr)

    opt = optax.chain(
        optax.scale(1 / gradient_accumulation_steps),
        clip_by_global_norm(1, use_psum=(version == 1)),
        optax.scale_by_adam(),
        additive_weight_decay(weight_decay),
        optax.scale(-1),
        optax.scale_by_schedule(scheduler)
    )

    params["optimizer"] = opt
    params["sampler"] = nucleaus_sample

    if version == 2:
        model_fn = functools.partial(CausalTransformerV2, params)
    elif version == 1:
        model_fn = functools.partial(CausalTransformer, params)
    else:
        raise Exception(f"Version {version} does not exist")

    t = TPUCluster((tpu_size // cores_per_replica, cores_per_replica), len(conns), model_fn, scheduler=scheduler, version=version)
    return t

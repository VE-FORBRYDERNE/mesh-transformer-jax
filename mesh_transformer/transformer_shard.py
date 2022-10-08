import gc
import random
import time
from functools import partial, reduce
from typing import Dict

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
try:
    import optax
    HAS_OPTAX = True
except ImportError:
    HAS_OPTAX = False
from jax.experimental.maps import thread_resources
from jax.experimental.pjit import pjit

from mesh_transformer.checkpoint import read_ckpt, write_ckpt, write_ckpt_v2, load_ckpt_v2
from mesh_transformer.layers import EmbeddingShard, TransformerLayerShard, RelativePositionEmbs, ProjectionShard, \
    create_alibi_tensor, \
    TransformerLayerShardV2, Projection, EmbeddingShardV2
from mesh_transformer.util import to_f32, to_bf16, maybe_shard, head_print, global_norm, f_psum
from mesh_transformer.util import process_index, process_count
from jax.experimental import PartitionSpec as P

from smart_open import open
import functools

if not getattr(print, "_train_actor_patched", False):
    old_print = print
    @functools.wraps(print)
    def new_print(*args, **kwargs):
        old_print(f"[NODE {process_index()}]", *args, **kwargs)
    new_print._train_actor_patched = True
    print = new_print


class PlaceholderTensor:
    def __init__(self, *shape: int):
        self.shape = shape
        self.size = reduce(lambda x, y: x * y, shape, 1)

    def __repr__(self):
        return type(self).__name__ + repr(self.shape)

    def __str__(self):
        return type(self).__name__ + str(self.shape)


def _create_dict(**kwargs):
    d = {}
    for k, v in kwargs.items():
        if v is not None:
            d[k] = v
    return d


def compute_placeholder_params(config: dict):
    compat = config.get("compat", "j")
    pe = config["pe"]
    use_combined_qkv = config.get("combined_qkv", compat in ("neox", "bloom"))
    do_layer_norm_before = config.get("do_layer_norm_before", True)

    if compat not in ("j", "neo", "fairseq_lm", "neox", "opt", "bloom"):
        raise NotImplementedError(f"Unsupported model type {repr(compat)}")
    if pe not in ("rotary", "neox_rotary", "fixed", "sinusoidal", "fairseq_sinusoidal", "alibi", "t5"):
        raise NotImplementedError(f"Unsupported positional embedding type {repr(pe)}")

    params: Dict[str, Dict[str, PlaceholderTensor]] = {}
    seq = config["seq"]
    in_dim = config["n_vocab"] + config.get("n_vocab_padding", 0)
    out_dim = config["d_model"]
    d_embed = config.get("d_embed", out_dim)
    shards = config["cores_per_replica"]
    in_dim_per_shard = in_dim // shards
    out_dim_per_shard = out_dim // shards
    ffn_dim_per_shard = out_dim_per_shard * 4

    if config["pe"] == "fixed" or d_embed != out_dim:
        params["causal_transformer_shard/~/embedding_shard"] = _create_dict(
            pos_embs=PlaceholderTensor(shards, seq, out_dim_per_shard) if config["pe"] == "fixed" else None,  # positional_embeddings
            project_in=PlaceholderTensor(shards, d_embed, out_dim_per_shard) if d_embed != out_dim else None,
        )

    params["causal_transformer_shard/~/embedding_shard/~/linear"] = _create_dict(  # proj
        w=PlaceholderTensor(shards, in_dim_per_shard, d_embed),
        b=PlaceholderTensor(shards, d_embed) if compat == "j" else None,
    )

    if compat == "bloom":
        params["causal_transformer_shard/~/embedding_shard/~/replicated_layer_norm"] = _create_dict(  # norm
            offset=PlaceholderTensor(shards, out_dim),
            scale=PlaceholderTensor(shards, out_dim),
        )

    for layer in range(config["layers"]):
        header = f"causal_transformer_shard/~/layer_{layer}/~/"
        if use_combined_qkv:
            params[header + "combined_qkv"] = _create_dict(  # qkv
                w=PlaceholderTensor(shards, out_dim, out_dim_per_shard * 3),
                b=PlaceholderTensor(shards, out_dim_per_shard * 3) if compat in ("fairseq_lm", "neox", "opt", "bloom") else None,
            )
        else:
            for footer in ("linear", "linear_1", "linear_2"):  # q, v, k
                params[header + footer] = _create_dict(
                    w=PlaceholderTensor(shards, out_dim, out_dim_per_shard),
                    b=PlaceholderTensor(shards, out_dim_per_shard) if compat in ("fairseq_lm", "neox", "opt", "bloom") else None,
                )
        params[header + "linear_3"] = _create_dict(  # o
            w=PlaceholderTensor(shards, out_dim_per_shard, out_dim),
            b=PlaceholderTensor(shards, out_dim) if compat in ("neo", "fairseq_lm", "neox", "opt", "bloom") else None,
        )
        params[header + "linear_4"] = _create_dict(  # dense_proj
            w=PlaceholderTensor(shards, out_dim, ffn_dim_per_shard),
            b=PlaceholderTensor(shards, ffn_dim_per_shard),
        )
        params[header + "linear_5"] = _create_dict(  # dense_proj_o
            w=PlaceholderTensor(shards, ffn_dim_per_shard, out_dim),
            b=PlaceholderTensor(shards, out_dim),
        )
        params[header + "replicated_layer_norm"] = _create_dict(  # norm
            offset=PlaceholderTensor(shards, out_dim),
            scale=PlaceholderTensor(shards, out_dim),
        )
        if compat != "j":
            params[header + "replicated_layer_norm_1"] = _create_dict(  # norm_2
                offset=PlaceholderTensor(shards, out_dim),
                scale=PlaceholderTensor(shards, out_dim),
            )

    if d_embed != out_dim:
        params["causal_transformer_shard/~/projection_shard"] = _create_dict(
            project_out=PlaceholderTensor(shards, out_dim, d_embed // shards)
        )
    if compat not in ("neo", "fairseq_lm", "opt", "bloom"):
        params["causal_transformer_shard/~/projection_shard/~/linear"] = _create_dict(  # proj
            w=PlaceholderTensor(shards, d_embed, in_dim_per_shard),
            b=PlaceholderTensor(shards, in_dim_per_shard) if compat == "j" else None,
        )
    if do_layer_norm_before or compat != "opt":
        params["causal_transformer_shard/~/projection_shard/~/replicated_layer_norm"] = _create_dict(  # norm
            offset=PlaceholderTensor(shards, out_dim),
            scale=PlaceholderTensor(shards, out_dim),
        )

    return params


class CausalTransformerShard(hk.Module):
    def __init__(self, config):
        super().__init__()
        heads = config["n_heads"]
        shards = config["cores_per_replica"]
        layer_count = config["layers"]
        self.vocab_size = config["n_vocab"]
        self.compat = config.get("compat", "j")

        self.transformer_layers = []
        self.heads = heads

        self.heads_per_shard = heads // shards

        self.embed = EmbeddingShard(config)

        self.pe = config["pe"]
        self.seq = config["seq"]

        init_scale = 2. / layer_count

        attention_layers = config.get("attention_layers", ["global" if self.compat != "neo" or i % 2 == 0 else "local" for i in range(config["layers"])])
        for i in range(layer_count):
            self.transformer_layers.append(TransformerLayerShard(config, name=f"layer_{i}", init_scale=init_scale, attention_type=attention_layers[i]))

        self.proj = ProjectionShard(config, embedding_shard=self.embed)

        if config["pe"] == "t5":
            self.rpe = RelativePositionEmbs()
        else:
            self.rpe = None

    def eval(self, context, target, z_loss=0., mask=0.0):
        input_len = context.shape[0]

        if self.rpe is not None:
            attn_bias = self.rpe(input_len, input_len, self.heads_per_shard, 32)
        elif self.pe == "alibi":
            attn_bias = create_alibi_tensor(self.heads, self.heads_per_shard, input_len)
        else:
            attn_bias = 0

        attn_bias += mask

        x = hk.remat(self.embed)(context, pe_length=input_len)

        for l in self.transformer_layers:
            x = x + hk.remat(l)(x, attn_bias)
            if not l.do_layer_norm_before:
                x = f_psum(x)
                x = hk.remat(l.norm)(x)
            if not l.neox_gpt_j_residual and l.compat != "j":
                x = x + hk.remat(l.neo_ff)(x)
                if not l.do_layer_norm_before:
                    x = f_psum(x)
                    x = hk.remat(l.norm_2)(x)

        return hk.remat(self.proj.loss)(x, target, z_loss)

    def loss(self, ctx, tgt, z_loss=False, mask=0.0):
        loss, correct = self.eval(ctx, tgt, float(z_loss), mask=mask)

        assert loss.ndim == 1
        return {
            "loss": loss.sum() / (tgt < self.vocab_size).sum(),
            "last_loss": loss[-1],
            "all_loss": loss,
            "correct": correct
        }

    def generate_initial(self, context, length, soft_embeddings=None):
        # slice last token off the context (we use that in generate_once to generate the first new token)
        last = context[-1:]
        context = context[:-1]

        input_len = context.shape[0]

        if self.rpe is not None:
            attn_bias = self.rpe(input_len, input_len, self.heads_per_shard, 32)
        elif self.pe == "alibi":
            attn_bias = create_alibi_tensor(self.heads, self.heads_per_shard, input_len)
        else:
            attn_bias = 0

        x = self.embed(context, pe_length=length - 1, soft_embeddings=soft_embeddings)

        states = []

        for l in self.transformer_layers:
            res, layer_state = l.get_init_decode_state(x, length - 1, attn_bias)
            x = x + res
            if not l.do_layer_norm_before:
                x = f_psum(x)
                x = l.norm(x)
            if not l.neox_gpt_j_residual and l.compat != "j":
                x = x + l.neo_ff(x)
                if not l.do_layer_norm_before:
                    x = f_psum(x)
                    x = l.norm_2(x)
            states.append(layer_state)

        return self.proj(x), (last.astype(jnp.uint32), states, hk.next_rng_key())

    def generate_once(self, new_tok, state, soft_embeddings=None):
        input_len = state[0]["v"].shape[0]

        if self.rpe is not None:
            attn_bias = self.rpe(input_len, input_len, self.heads_per_shard, 32)
            attn_bias = attn_bias[:, -1:, :]
        elif self.pe == "alibi":
            attn_bias = create_alibi_tensor(self.heads, self.heads_per_shard, input_len)
        else:
            attn_bias = 0

        x = self.embed(new_tok, pe_length=state[0]["tokens_decoded"] + 1, soft_embeddings=soft_embeddings)

        new_states = []

        for l, s in zip(self.transformer_layers, state):
            res, layer_state = l.decode_once(s, x, attn_bias)
            x = x + res
            if not l.do_layer_norm_before:
                x = f_psum(x)
                x = l.norm(x)
            if not l.neox_gpt_j_residual and l.compat != "j":
                x = x + l.neo_ff(x)
                if not l.do_layer_norm_before:
                    x = f_psum(x)
                    x = l.norm_2(x)
            new_states.append(layer_state)

        return self.proj(x), new_states


class CausalTransformer:
    def __init__(self, config, dematerialized=False):
        self.config = config
        optimizer = config["optimizer"]

        def eval(state, ctx, tgt, ctx_length):
            def eval_loss(x, y, mask):
                transformer = CausalTransformerShard(config)
                return transformer.loss(x, y, mask=mask)

            eval_loss_fn = hk.without_apply_rng(hk.transform(eval_loss)).apply

            mask = (jnp.arange(0, len(ctx)) > ctx_length) * -1e10

            return eval_loss_fn(to_bf16(state["params"]), ctx, tgt, mask)

        def train(state, ctx, tgt):
            def train_loss(x, y):
                transformer = CausalTransformerShard(config)
                out = transformer.loss(x, y, z_loss=True)

                return out["loss"], out["last_loss"]

            train_loss_fn = hk.without_apply_rng(hk.transform(train_loss)).apply

            def microbatch(old_grad, batch):
                ctx, tgt = batch

                val_grad_fn = jax.value_and_grad(train_loss_fn, has_aux=True)
                (loss, last_loss), grad = val_grad_fn(to_bf16(state["params"]), ctx, tgt)

                new_grad = jax.tree_multimap(lambda a, b: a + b, old_grad, grad)
                gnorm = global_norm(grad)
                return new_grad, (loss, last_loss, gnorm)

            if ctx.shape[0] == 1:
                val_grad_fn = jax.value_and_grad(train_loss_fn, has_aux=True)
                (loss, last_loss), grad = val_grad_fn(to_bf16(state["params"]), ctx[0], tgt[0])
                gnorm = global_norm(grad)
            else:
                grad, (loss, last_loss, gnorm) = jax.lax.scan(microbatch,
                                                       jax.tree_map(lambda x: jnp.zeros_like(x).astype(jnp.bfloat16),
                                                                    state["params"]),
                                                       (ctx, tgt))

            grad_norm_micro = jax.lax.pmean(gnorm, "batch")

            grad = jax.lax.psum(grad, "batch")
            grad_norm = global_norm(grad)
            updates, new_opt_state = optimizer.update(grad, state["opt_state"], state["params"])

            return to_f32(loss), to_f32(last_loss), to_f32(grad_norm), to_f32(grad_norm_micro), {
                "params": optax.apply_updates(state["params"], to_f32(updates)) if HAS_OPTAX else None,
                "step": state["step"] + 1,
                "opt_state": new_opt_state
            }

        def init(key, x):
            def train_loss(x, y):
                transformer = CausalTransformerShard(config)
                return transformer.loss(x, y)

            param_init_fn = hk.transform(hk.experimental.optimize_rng_use(train_loss)).init

            params = param_init_fn(key, x, x)

            return {
                "params": ("early_cast" in config and to_bf16 or to_f32)(params),
                "step": np.array(0),
                "opt_state": optimizer.init(params)
            }

        def generate(state, key, ctx, ctx_length, aux, sampler_options, soft_embeddings=None):
            sampler = config["sampler"]
            gen_length = self.gen_length

            def generate_sample(context, ctx_length, aux):
                transformer = CausalTransformerShard(config)
                _, initial_state = transformer.generate_initial(context, ctx_length, soft_embeddings=soft_embeddings)

                def generate_scan_fn(carry, sampler_input):
                    next_token, decode_state, sample_key = carry
                    sample_key, new_key = jax.random.split(sample_key)

                    logits, new_state = transformer.generate_once(next_token, decode_state, soft_embeddings=soft_embeddings)
                    next_token, sample_info = sampler(sample_key, logits, sampler_input, **sampler_options)

                    if self.return_logits:
                        output = (next_token, sample_info, logits)
                    else:
                        output = (next_token, sample_info)
                    new_carry = (next_token, new_state, new_key)
                    return new_carry, output

                final_state, outputs = jax.lax.scan(generate_scan_fn, initial_state, xs=aux, length=gen_length)
                return final_state, outputs

            generate_fn = hk.transform(generate_sample).apply
            return generate_fn(state["params"], key, ctx, ctx_length, aux)

        self.init_xmap = jax.experimental.maps.xmap(fun=init,
                                                    in_axes=(["shard", ...],
                                                             ["batch", ...]),
                                                    out_axes=["shard", ...],
                                                    axis_resources={'shard': 'mp', 'batch': 'dp'})

        self.eval_xmap = jax.experimental.maps.xmap(fun=eval,
                                                    in_axes=(["shard", ...],
                                                             ["batch", ...],
                                                             ["batch", ...],
                                                             ["batch", ...]),
                                                    out_axes=["batch", ...],
                                                    axis_resources={'shard': 'mp', 'batch': 'dp'})

        self.train_xmap = jax.experimental.maps.xmap(fun=train,
                                                     in_axes=(["shard", ...],
                                                              ["batch", ...],
                                                              ["batch", ...]),
                                                     out_axes=(["batch", ...], ["batch", ...], ["batch", ...], ["batch", ...], ["shard", ...]),
                                                     donate_argnums=(0,),
                                                     axis_resources={'shard': 'mp', 'batch': 'dp'})

        self.generate_xmap = jax.experimental.maps.xmap(fun=generate,
                                                        in_axes=(["shard", ...],
                                                                 ["batch", ...],
                                                                 ["batch", ...],
                                                                 ["batch", ...],
                                                                 ["batch", ...],
                                                                 ["batch", ...],
                                                                 ["shard", ...]),
                                                        out_axes=["batch", ...],
                                                        axis_resources={'shard': 'mp', 'batch': 'dp'})

        self.move_xmap = jax.experimental.maps.xmap(fun=lambda x, _: to_bf16(x),
                                                    in_axes=(["shard", ...], ["batch", ...]),
                                                    out_axes=["shard", ...],
                                                    axis_resources={'shard': 'mp', 'batch': 'dp'})

        key = hk.PRNGSequence(42)

        assert thread_resources.env.shape['mp'] == config["cores_per_replica"]

        dp = thread_resources.env.shape['dp']
        mp = thread_resources.env.shape['mp']

        mp_per_host = min(mp, 8)

        seq = config["seq"]
        vocab = config["n_vocab"] + config.get("n_vocab_padding", 0)

        example_shape = (max(dp // jax.host_count(), 1), seq,)
        x = jax.random.uniform(next(key), example_shape, minval=0, maxval=vocab).astype(jnp.uint32)  # batch, len

        head_print("key shape", jnp.array(key.take(mp_per_host)).shape)
        head_print("in shape", x.shape)

        head_print("dp", dp)
        head_print("mp", mp)

        self.gen_length = 1
        self.state = self.init_xmap(jnp.array(key.take(mp_per_host)), x) if not dematerialized else {
            "params": compute_placeholder_params(config),
            "step": jnp.zeros(mp, dtype=jnp.uint32),
            "opt_state": optimizer.init({})
        }

        param_count = hk.data_structures.tree_size(self.state['params'])
        head_print(f"Total parameters: {param_count}")

    def write_ckpt(self, path, shard, save_opt_state=True):
        node_index = process_index()
        if node_index >= thread_resources.env.shape['mp'] // 8:
            print(f"Skipping saving for node {node_index} because {node_index} >= mp/8")
            return
        total_shards = self.config.get("shards", self.config["cores_per_replica"])
        print(f"Node {node_index} is saving shard {shard + (node_index * jax.local_device_count())%total_shards} to {path}")
        write_ckpt(self.state, path, shard, node_index, save_opt_state=save_opt_state, total_shards=total_shards)

    def load_ckpt(self, path, finetuning=False):
        node_index = process_index()
        shard_offset = (node_index * jax.local_device_count()) % self.config.get("shards", self.config["cores_per_replica"])
        print(f"Node {node_index} is allocated shards {shard_offset}-{shard_offset + jax.local_device_count() - 1}")
        init_sched_state = self.state["opt_state"][-1]
        self.state = read_ckpt(self.state, path, shards_in=min(jax.local_device_count(), self.config.get("shards", thread_resources.env.shape['mp'])), shards_out=min(jax.local_device_count(), thread_resources.env.shape['mp']), shard_offset=shard_offset)
        if finetuning:
            self.state["opt_state"][-1] = init_sched_state

    def train(self, sample, chunks=1):
        # print("train iter")
        # print("sample", sample["obs"])
        # print("target", sample["target"])

        sample_obs = jnp.array_split(sample["obs"], chunks, axis=1)
        sample_obs = sample_obs[process_index() // (process_count() // chunks)]
        sample_target = jnp.array_split(sample["target"], chunks, axis=1)
        sample_target = sample_target[process_index() // (process_count() // chunks)]

        obs = jnp.transpose(sample_obs, (1, 0, 2))
        target = jnp.transpose(sample_target, (1, 0, 2))

        # print("train sample", obs.shape)
        # print("train target", target.shape)

        # start = time.time()
        loss, last_loss, grad_norm, grad_norm_micro, self.state = self.train_xmap(self.state, obs, target)
        loss = np.array(loss)
        last_loss = np.array(last_loss)
        grad_norm = np.array(grad_norm)
        # print(f"iter done in {time.time() - start:.06}s")
        return loss.mean(), last_loss.mean(), grad_norm.mean(), grad_norm_micro.mean()

    def eval(self, sample):
        # print("eval sample", sample["obs"].shape)
        # print("eval target", sample["target"].shape)

        # start = time.time()

        if "ctx_length" in sample:
            ctx_length = sample["ctx_length"]
        else:
            ctx_length = np.array([len(sample["obs"][0])] * len(sample["obs"]))

        out = self.eval_xmap(self.state, sample["obs"], sample["target"], ctx_length)
        # print(f"eval dispatched in {time.time() - start:.06}s")

        # np.array(out["loss"])
        # print(f"eval done in {time.time() - start:.06}s")
        return out

    def generate(self, ctx, ctx_length, gen_length, sampler_options, return_logits=False, soft_embeddings=None):
        key = hk.PRNGSequence(random.randint(0, 2 ** 60))

        batch_size = ctx.shape[0]
        aux = jnp.zeros((batch_size, gen_length), dtype=jnp.uint32)
        self.gen_length = gen_length
        self.return_logits = return_logits

        return self.generate_xmap(self.state,
                                  jnp.array(key.take(batch_size)),
                                  ctx,
                                  np.array(ctx_length, dtype=np.uint32),
                                  aux,
                                  sampler_options,
                                  soft_embeddings)


# this bypasses the CausalTransformerShard class (which causes ugly code) but in return allows layers to be processed
# by a `jax.scan`, which allows for much faster and O(1) compile times w.r.t. layers.
class CausalTransformerV2:
    def __init__(self, config):
        self.config = config
        optimizer = config["optimizer"]
        with_optimizer = optimizer is not None

        head_print(f"with_optimizer: {with_optimizer}")

        bf16_optimizer = config.get("bf16_optimizer", False)
        early_cast = config.get("early_cast", False)
        early_collect = config.get("early_collect", True)

        def embedding(x):
            x = maybe_shard(x, P("dp", None))
            return EmbeddingShardV2(config)(x)

        def residual(x, mask):
            out = x + TransformerLayerShardV2(config, init_scale=2. / config["layers"])(x, mask)
            return maybe_shard(out, P("dp", None, "mp"))

        def init_decode(x, given_length, mask):
            residual, decode_state = TransformerLayerShardV2(config, init_scale=2. / config["layers"])\
                .get_init_decode_state(x, given_length, mask)
            out = x + residual
            return maybe_shard(out, P("dp", None, "mp")), decode_state

        def iter_decode(decode_state, x):
            residual, decode_state = TransformerLayerShardV2(config, init_scale=2. / config["layers"])\
                .decode_once(decode_state, x, 0)
            out = x + residual
            return maybe_shard(out, P("dp", None, "mp")), decode_state

        def transformer(x, mask):
            return hk.remat(residual, prevent_cse=False)(x, mask)

        def projection(x):
            return Projection(config)(x)

        def init_fns():
            embed_init_fn = hk.transform(hk.experimental.optimize_rng_use(embedding)).init
            transformer_init_fn = hk.transform(hk.experimental.optimize_rng_use(transformer)).init
            projection_init_fn = hk.transform(hk.experimental.optimize_rng_use(projection)).init

            return embed_init_fn, transformer_init_fn, projection_init_fn

        def shard_strategy(shape_dtype, parallel):
            if shape_dtype.ndim == 0:
                return P()
            if shape_dtype.ndim == 1:
                return P(None)
            # embedding/projection layers
            elif shape_dtype.shape == (config["n_vocab"], config["d_model"]):
                return P(parallel, None)
            elif shape_dtype.shape == (config["d_model"], config["n_vocab"]):
                return P(None, parallel)

            # a transformer layer
            elif shape_dtype.shape[0] == config["layers"]:
                if shape_dtype.ndim == 2:
                    # a channel wise variable (e.g. layernorm parameters)
                    # replicate it for speed
                    return P(None, None)
                elif shape_dtype.ndim == 3:
                    # a weight matrix
                    matrix_size = shape_dtype.shape[1:]

                    assert matrix_size[0] != matrix_size[1]  # this case is ambiguous

                    if matrix_size[0] == config["d_model"]:
                        # shard along the axis which is _not_ the model dimension
                        return P(None, None, parallel)
                    elif matrix_size[1] == config["d_model"]:
                        return P(None, parallel, None)
                else:
                    raise NotImplementedError("borked")

            else:
                raise NotImplementedError("borked")

        def init(key, x):
            embed_init_fn, transformer_init_fn, projection_init_fn = init_fns()

            def init_scan_fn(key, x):
                new_key, key = jax.random.split(key)

                return new_key, transformer_init_fn(key, x, 0)

            e_key, t_key, p_key = jax.random.split(key, 3)

            input_shape = (config["layers"],) + x.shape + (config["d_model"],)

            params = {
                "embed": embed_init_fn(e_key, x),
                "transformer": jax.lax.scan(init_scan_fn,
                                            t_key,
                                            xs=jax.random.uniform(t_key, input_shape, dtype=jnp.float32))[1],
                "proj": projection_init_fn(p_key, jax.random.uniform(t_key, input_shape[1:], dtype=jnp.float32)),
            }

            output_state = {
                "params": (to_bf16 if early_cast else to_f32)(params),
                "step": np.array(0),
            }

            if with_optimizer:
                output_state["opt_state"] = optimizer.init((to_bf16 if bf16_optimizer else to_f32)(params))

            return output_state

        assert thread_resources.env.shape['mp'] == config["cores_per_replica"]

        dp = thread_resources.env.shape['dp']
        mp = thread_resources.env.shape['mp']

        key = hk.PRNGSequence(42)
        x = jax.random.uniform(next(key), (mp * dp, 16), minval=0, maxval=1).astype(jnp.uint32)  # batch, seq

        head_print("starting shape evaluation")

        param_shapes = jax.eval_shape(init, jax.random.PRNGKey(42), x)

        state_shard = {
                "step": P(),

                # fp32 params are also sharded (so this is like a weird mix between zero-1 and zero-3...)
                "params": jax.tree_map(partial(shard_strategy, parallel=["mp", "dp"]), param_shapes["params"]),
            }

        if "opt_state" in param_shapes:
            # zero level 1: shard optimizer states over both MP and DP
            state_shard["opt_state"] = jax.tree_map(partial(shard_strategy, parallel=["mp", "dp"]), param_shapes["opt_state"])

        self.state_shard = state_shard

        head_print("sharding strategy:")
        # head_print("state shard: ", state_shard)
        # head_print("param_shapes: ", param_shapes)
        jax.tree_multimap(head_print, state_shard, param_shapes)

        self.init_pjit = pjit(init, in_axis_resources=(None, P("dp")), out_axis_resources=state_shard)

        def apply_fns():
            embed_apply_fn = hk.without_apply_rng(hk.transform(embedding)).apply
            transformer_apply_fn = hk.without_apply_rng(hk.transform(transformer)).apply

            return embed_apply_fn, transformer_apply_fn

        def train_apply_fn(params, x, y):
            embed_apply_fn, transformer_apply_fn = apply_fns()

            def train_loss(x, y):
                loss, _ = Projection(config).loss(x, y, z_loss=1.0)
                return loss.mean(), loss[:, -1].mean()

            projection_apply_fn = hk.without_apply_rng(hk.transform(train_loss)).apply

            x = embed_apply_fn(params["embed"], x)
            x = to_bf16(x)

            def apply_scan_fn(x, layer_state):
                return to_bf16(transformer_apply_fn(layer_state, x, 0)), None

            x = jax.lax.scan(apply_scan_fn,
                             x,
                             xs=params["transformer"])[0]

            return projection_apply_fn(params["proj"], x, y)

        mp_shard_strategy = jax.tree_map(partial(shard_strategy, parallel=["mp"]), param_shapes["params"])

        def train(state, ctx, tgt):
            if early_collect:
                bf16_params = maybe_shard(to_bf16(state["params"]), mp_shard_strategy)
            else:
                bf16_params = to_bf16(state["params"])

            def microbatch(old_grad, batch):
                ctx, tgt = batch

                val_grad_fn = jax.value_and_grad(train_apply_fn, has_aux=True, allow_int=True)
                (loss, last_loss), grad = val_grad_fn(bf16_params, ctx, tgt)

                new_grad = jax.tree_multimap(lambda a, b: a + b, old_grad, grad)
                return new_grad, (loss, last_loss)

            if ctx.shape[0] == 1:
                val_grad_fn = jax.value_and_grad(train_apply_fn, has_aux=True, allow_int=True)
                (loss, last_loss), grad = val_grad_fn(bf16_params, ctx[0], tgt[0])
            else:
                grad, (loss, last_loss) = jax.lax.scan(microbatch,
                                                       jax.tree_map(lambda x: jnp.zeros_like(x).astype(jnp.bfloat16),
                                                                    bf16_params),
                                                       (ctx, tgt))

            updates, new_opt_state = optimizer.update(grad, state["opt_state"], state["params"])

            return to_f32(loss), to_f32(last_loss), {
                "params": optax.apply_updates(state["params"], to_f32(updates)) if HAS_OPTAX else None,
                "step": state["step"] + 1,
                "opt_state": new_opt_state,
            }

        self.train_pjit = pjit(train,
                               in_axis_resources=(state_shard, P(None, "dp"), P(None, "dp")),
                               out_axis_resources=(None, None, state_shard),
                               donate_argnums=(0,))

        def eval_apply_fn(params, x, y, mask):
            embed_apply_fn, transformer_apply_fn = apply_fns()

            if early_collect:
                bf16_params = maybe_shard(to_bf16(params), mp_shard_strategy)
            else:
                bf16_params = to_bf16(params)

            def eval_loss(x, y):
                loss, correct = Projection(config).loss(x, y)
                return {
                            "loss": loss.mean(axis=-1),
                            "last_loss": loss[:, -1],
                            "all_loss": loss,
                            "correct": correct
                        }

            projection_apply_fn = hk.without_apply_rng(hk.transform(eval_loss)).apply

            x = embed_apply_fn(bf16_params["embed"], x)

            def apply_scan_fn(layer_in, layer_state):
                x, mask = layer_in
                return (to_bf16(transformer_apply_fn(layer_state, x, mask)), mask), None

            x = jax.lax.scan(apply_scan_fn,
                             (to_bf16(x), mask),
                             xs=bf16_params["transformer"])[0][0]

            return projection_apply_fn(bf16_params["proj"], x, y)

        def eval(params, ctx, tgt, ctx_length):
            mask = (jnp.arange(0, ctx.shape[1])[None, :] > ctx_length[:, None]) * -1e10

            # head_print("mask.shape", mask.shape)
            # head_print("ctx.shape", ctx.shape)
            # head_print("ctx_length.shape", ctx_length.shape)

            return eval_apply_fn(params, ctx, tgt, mask[:, None, None, :])

        self.eval_pjit = pjit(eval,
                              in_axis_resources=(mp_shard_strategy if early_collect else state_shard["params"],
                                                 P("dp"), P("dp"), P("dp")),
                              out_axis_resources=P("dp"))

        def generate(params, key, ctx, ctx_length, aux, sampler_options):
            sampler = config["sampler"]
            gen_length = config["gen_length"]

            embed_apply_fn, _ = apply_fns()
            init_decode_apply = hk.without_apply_rng(hk.transform(init_decode)).apply
            iter_decode_apply = hk.without_apply_rng(hk.transform(iter_decode)).apply

            def get_inital(params, ctx, ctx_length):
                x = embed_apply_fn(params["embed"], ctx)
                mask = (jnp.arange(0, ctx.shape[1])[None, :] > ctx_length[:, None]) * -1e10

                def apply_scan_fn(layer_in, layer_state):
                    x, mask = layer_in

                    x, decode_state = init_decode_apply(layer_state, x, mask)
                    return (x, mask), decode_state

                _, init_state = jax.lax.scan(apply_scan_fn,
                                 (to_bf16(x), mask),
                                 xs=params["transformer"])

                return (last.astype(jnp.uint32), init_state, hk.next_rng_key())

            initial_state = get_inital(params, ctx, ctx_length)
            initial_carry = ()

            def generate_scan_fn(carry, sampler_input):
                next_token, decode_state, sample_key = carry
                sample_key, new_key = jax.random.split(sample_key)

                x = embed_apply_fn(params["embed"], next_token)
                mask = (jnp.arange(0, ctx.shape[1])[None, :] > ctx_length[:, None]) * -1e10

                def layer_scan_fn(carry_in, layer_in):
                    x, mask = carry_in
                    layer_state, decode_state = layer_in

                    x, decode_state = iter_decode_apply(layer_state, decode_state, x)
                    return (x, mask), decode_state

                (x, _), new_state = jax.lax.scan(layer_scan_fn,
                                             (to_bf16(x), mask),
                                             xs=params["transformer"])

                projection_apply_fn = hk.without_apply_rng(hk.transform(Projection(config))).apply

                logits = projection_apply_fn(params["proj"], x)
                next_token, sample_info = sampler(sample_key, logits, sampler_input, **sampler_options)

                new_carry = (next_token, new_state, new_key)
                return new_carry, (next_token, sample_info)

            final_state, outputs = jax.lax.scan(generate_scan_fn, initial_state, xs=aux, length=gen_length)
            return final_state, outputs

        self.move_weights_pjit = pjit(lambda x: to_bf16(x),
                                      in_axis_resources=(state_shard["params"], ),
                                      out_axis_resources=mp_shard_strategy if early_collect else state_shard["params"])

        seq = config["seq"]
        vocab = config["n_vocab"]

        example_shape = (max(dp // jax.host_count(), 1), seq,)
        x = jax.random.uniform(next(key), example_shape, minval=0, maxval=vocab).astype(jnp.uint32)  # batch, len

        head_print("in shape", x.shape)

        head_print("dp", dp)
        head_print("mp", mp)

        self.state = self.init_pjit(next(key), x)
        self.state_shard = state_shard

        if with_optimizer:
            self.eval_weights = None
        else:
            self.eval_weights = self.state["params"]

        param_count = hk.data_structures.tree_size(self.state['params'])
        head_print(f"Total parameters: {param_count * dp}")

    def write_ckpt(self, path, _, save_opt_state=True):
        write_ckpt_v2(self.state, path)

    def load_ckpt(self, path):
        self.state = load_ckpt_v2(self.state, path, self.state_shard, not self.config.get("eval_only", False))

    def train(self, sample):
        # print("train iter")
        # print("sample", sample["obs"])
        # print("target", sample["target"])

        obs = sample["obs"]
        target = sample["target"]

        if self.eval_weights is not None:
            self.eval_weights = None
            gc.collect()
            head_print("deleted eval weights")

        # print("train sample", obs.shape)
        # print("train target", target.shape)

        # assert (sample["obs"][:, 1:] == sample["target"][:, -1])

        start = time.time()
        loss, last_loss, self.state = self.train_pjit(self.state, obs, target)
        loss = np.array(loss)
        last_loss = np.array(last_loss)

        # head_print(f"iter done in {time.time() - start:.06}s")
        return loss.mean(), last_loss.mean()

    def eval(self, sample):
        # head_print("eval sample", sample["obs"].shape)
        # print("eval target", sample["target"].shape)

        start = time.time()

        if self.eval_weights is None:
            self.eval_weights = self.move_weights_pjit(self.state["params"])

            # blocking
            jnp.zeros(()).block_until_ready()

            head_print(f"created eval weights in {time.time() - start:.06}s")

        if "ctx_length" in sample:
            ctx_length = sample["ctx_length"]
        else:
            ctx_length = np.array([len(sample["obs"][0])] * len(sample["obs"]))

        # head_print("ctx_length in eval", ctx_length)

        out = self.eval_pjit(self.eval_weights, sample["obs"], sample["target"], ctx_length)
        # print(f"eval dispatched in {time.time() - start:.06}s")

        # np.array(out["loss"])
        # print(f"eval done in {time.time() - start:.06}s")
        return out

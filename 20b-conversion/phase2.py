#!/usr/bin/env python3

import numpy as np
import jax.numpy as jnp
import os

total_shards = 32

lengths = [34, 34, 34, 34, 34, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33]

def reshard_reverse(x, old_shape, is_shard_bias=False):
    if len(x.shape) == 1:
        assert False
        out = x[0:1]

    elif len(x.shape) == 2:
        #print(f"LN/bias")
        if old_shape[1] == x.shape[1]:
            #print("LN")
            if not is_shard_bias:
                out = np.tile(x[0:1], (total_shards, 1))
            else:
                #print("shard bias")
                out = np.tile(x[0:1], (total_shards, 1)) / total_shards
        else:
            #print("bias")
            out = x.reshape(old_shape)

    elif len(x.shape) == 3:
        if x.shape[0] * x.shape[2] == old_shape[2]:
            #print("case 1")
            out = x.reshape(old_shape)
        elif x.shape[0] * x.shape[1] == old_shape[1]:
            #print("case 2")
            out = jnp.transpose(x.reshape((old_shape[1], old_shape[0], old_shape[2])), (1, 0, 2))
        else:
            raise Exception(f"unimplemented, {x.shape}, {old_shape}")
    else:
        raise Exception(f"unimplemented, {x}")
    #flattened, structure = jax.tree_flatten(out)
    #return flattened
    return out

def get_old_shape(t, dim=2):
    if len(t.shape) == 2:
        shard_shape = t.shape
        if dim == 1:
            assert shard_shape[0] % total_shards == 0
            return (shard_shape[0] // total_shards, shard_shape[1])
        elif dim == 2:
            assert shard_shape[1] % total_shards == 0
            return (shard_shape[0], shard_shape[1] // total_shards)
        else:
            raise ValueError(f"unsupported dim {dim}")
    if len(t.shape) == 1:
        assert t.shape[0] % total_shards == 0
        return (t.shape[0] // total_shards,)
    else:
        raise ValueError(f"unsupported shape {t.shape}")


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


transforms = [
    ("gpt_neox.embed_in.weight", False, 1)
]
layers = 44

checkpoint = []

layer_names = sorted(map(str, range(layers)))
for layer in layer_names:
    transforms.extend([
        (f"gpt_neox.layers.{layer}.attention.query_key_value.bias", False, 1),
        (f"gpt_neox.layers.{layer}.attention.query_key_value.weight", False, 2),
        (f"gpt_neox.layers.{layer}.attention.dense.bias", True, None),
        (f"gpt_neox.layers.{layer}.attention.dense.weight", False, 1),
        (f"gpt_neox.layers.{layer}.mlp.dense_h_to_4h.bias", False, 1),
        (f"gpt_neox.layers.{layer}.mlp.dense_h_to_4h.weight", False, 2),
        (f"gpt_neox.layers.{layer}.mlp.dense_4h_to_h.bias", True, None),
        (f"gpt_neox.layers.{layer}.mlp.dense_4h_to_h.weight", False, 1),
        (f"gpt_neox.layers.{layer}.input_layernorm.bias", False, None),
        (f"gpt_neox.layers.{layer}.input_layernorm.weight", False, None),
        (f"gpt_neox.layers.{layer}.post_attention_layernorm.bias", False, None),
        (f"gpt_neox.layers.{layer}.post_attention_layernorm.weight", False, None),
    ])
transforms.extend([
    ("embed_out.weight", False, 2),
    ("gpt_neox.final_layer_norm.bias", False, None),
    ("gpt_neox.final_layer_norm.weight", False, None),
])

file_index = 0

for i in range(len(transforms)):
    transform = transforms.pop(0)

    # params = torch_checkpoint["model"][transform[0]]
    params = np.load(f"phase1_out/{transform[0]}.npy")

    # if transform[0] in ("decoder.embed_tokens.weight", "decoder.output_projection.weight"):
    #     params = torch.cat((params, torch.zeros(padding_rows, params.shape[-1], device=params.device)), dim=0)

    # torch.nn.Linear uses a transposed version of the equivalent tensor that
    # haiku.Linear uses, so we have to un-transpose the tensor first
    if not any(s in transform[0] for s in ("gpt_neox.embed_in.weight",)):
        params = params.T

    if transform[2] is not None:
        old_shape = (total_shards,) + get_old_shape(params, transform[2])
    else:
        old_shape = (total_shards, params.shape[0],)
    print(f"< [{transform[0]}] {params.shape} to {old_shape}")

    params = np.array(params[None], dtype=jnp.bfloat16)
    params = reshard_reverse(params, old_shape, is_shard_bias=transform[1])

    if np.isnan(params).any() or np.isinf(params).any():
        raise ValueError(f"bfloat16 overflow/underflow")

    print(f"> [{transform[0]}] {params.shape}")
    assert params.shape == old_shape
    checkpoint.append(params)

    if len(checkpoint) == lengths[file_index]:
        print(f"! Saving chunk {file_index}")
        for i in range(total_shards):
            os.makedirs(f"phase2_out/step_150000/shard_{i}", exist_ok=True)
            with open(f"phase2_out/step_150000/shard_{i}/{file_index}.npz", "wb") as f:
                np.savez(f, *map(lambda c: c[i], checkpoint))
        checkpoint = []
        file_index += 1

checkpoint.append(np.ones(total_shards, dtype=np.int32) * 150000)

if len(checkpoint) == lengths[file_index]:
    print(f"! Saving chunk {file_index}")
    for i in range(total_shards):
        os.makedirs(f"phase2_out/step_150000/shard_{i}", exist_ok=True)
        with open(f"phase2_out/step_150000/shard_{i}/{file_index}.npz", "wb") as f:
            np.savez(f, *map(lambda c: c[i], checkpoint))
    checkpoint = []
    file_index += 1

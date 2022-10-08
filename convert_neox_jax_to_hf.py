import os
import argparse
import json
import time
import sys
import random

from argparse import ArgumentParser

import torch
import numpy as np
import jax.numpy as jnp
import io
from smart_open import open


# Ratio of v8 compared to total size
# pe_rotary_dims = 0.25 * (hidden_size // num_attention_heads)
pe_rotary_dims = 24

layers = 44
total_shards = 32
default_ckpt_dir = "gs://q00q/20B-32/step_154189/"
default_destination = "/"

def parseArgs():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", "--ckpt_dir", dest="ckpt_dir", default=default_ckpt_dir,
                    help="Location of the checkpoint to convert into HF model", metavar="FILE")
    parser.add_argument("-o", "--output", "--destination", dest="destination", default=default_destination,
                    help="Where to write the resulting HF model to.")
    parser.add_argument("-s", "--shards", "--total_shards", dest="total_shards", default=total_shards,
                    help="Number of shards in the original.")
    # todo: !! ftw parser.add_argument("-q", "--quiet", action="store_false", dest="verbose", default=True,
    #                 help="Don't be chatty.")
    args = parser.parse_args()

    return args

args = parseArgs()

ckpt_dir = args.ckpt_dir
destination = args.destination
total_shards = args.total_shards

print('args:', args)
print('destination:', destination)
print('ckpt_dir:', ckpt_dir)
print('total_shards:', total_shards)
print('———')

def reshard(x, old_shape, is_shard_bias=False):
    if len(x.shape) == 1:
        # print("epoch")
        # print(x)
        out = x[0:1]

    elif len(x.shape) == 2:
        #print(f"LN/bias {x.shape}")
        #print(x[:, :16])

        if old_shape[1] == x.shape[1]:
            #print("LN")
            if not is_shard_bias:#if (x[1:] == 0).all() or (x[1:] == 1).all():
                out = x[0:1]
            else:
                #print("shard bias")
                out = x[0:1] * total_shards#* x.shape[0] / old_shape[0]
        else:
            #print("bias")
            out = x.reshape(old_shape)

        #print(out[:, :16])

    elif len(x.shape) == 3:
        #print(f"weight {x.shape}")
        if x.shape[0] * x.shape[2] == old_shape[2]:
            #print("case 1")
            out = jnp.transpose(x, (1, 0, 2)).reshape(old_shape)
        elif x.shape[0] * x.shape[1] == old_shape[1]:
            #print("case 2")
            out = x.reshape(old_shape)
        else:
            raise Exception(f"unimplemented, {x.shape}, {old_shape}")
    else:
        raise Exception(f"unimplemented, {x}")
    #flattened, structure = jax.tree_flatten(out)
    #return flattened
    return out

def get_old_shape(t, dim=2):
    if len(t.shape) == 3:
        shard_shape = t.shape
        if dim == 1:
            return (shard_shape[0] * shard_shape[1], shard_shape[2])
        elif dim == 2:
            return (shard_shape[1], shard_shape[0] * shard_shape[2])
        else:
            raise ValueError(f"unsupported dim {dim}")
    if len(t.shape) == 2:
        return (t.shape[1] * t.shape[0],)
    else:
        raise ValueError(f"unsupported shape {t.shape}")

def read_shard(ckpt_dir):
    global part
    out = []
    idx = part
    file_path = ckpt_dir + f"{idx}.npz"
    #print(f"-- {file_path}")
    with open(file_path, "rb") as f:
        buf = f.read()
        f_io = io.BytesIO(buf)
        deserialized = np.load(f_io)
        for i in deserialized:
            out.append(deserialized[i])
            #print(deserialized[i].shape)
    return out

unshard = None
transforms = [("gpt_neox.embed_in.weight", False, 1)]

checkpoint = {}

layer_names = sorted(map(str, range(layers)))
for layer in layer_names:
    checkpoint[f"gpt_neox.layers.{layer}.attention.bias"] = torch.tril(torch.ones(1, 1, 2048, 2048, dtype=torch.uint8))
    checkpoint[f"gpt_neox.layers.{layer}.attention.masked_bias"] = torch.tensor(-1e9, dtype=torch.float16)
    checkpoint[f"gpt_neox.layers.{layer}.attention.rotary_emb.inv_freq"] = (1.0 / (10000 ** (torch.arange(0, pe_rotary_dims, 2).float() / pe_rotary_dims))).to(torch.float16)
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

part = 0
element = 0
while len(transforms) > 0:
    print(f"loading shards for part {part}")
    shards = list(map(read_shard, [f"{ckpt_dir}shard_{i}/" for i in range(total_shards)]))
    print(f"read from checkpoint")

    unsharded = []

    for all_shards in zip(*shards):
        x = np.stack(all_shards)
        # No idea why this is V2...?
        if x.dtype == np.dtype('V2'):
            x.dtype = jnp.bfloat16
        x = x.astype(np.float32)
        unsharded.append(x)
        #print(f"unsharded: {x.shape}")

    while len(transforms) > 0 and len(unsharded) > 0:
        transform = transforms.pop(0)
        params = unsharded.pop(0)
        if transform[2] is not None:
            old_shape = (1,) + get_old_shape(params, transform[2])
        else:
            old_shape = (1,) + (params.shape[1],)
        print(f"< {params.shape} to {old_shape}")
        params = reshard(params, old_shape, is_shard_bias=transform[1]).squeeze(0).T
        params = torch.tensor(params.copy()).half()
        if params.isnan().any() or params.isinf().any():
            raise ValueError(f"fp16 over/underflow at {part} {element}")
        checkpoint[transform[0]] = params
        print(f"> {transform[0]} {params.shape}")
        element += 1
    part += 1

checkpoint['gpt_neox.embed_in.weight'] = checkpoint['gpt_neox.embed_in.weight'].T

rng = random.Random(1729)
os.makedirs(destination, exist_ok=True)
checkpoint = list(checkpoint.items())
rng.shuffle(checkpoint)

import transformers.modeling_utils
max_shard_size = 2 * 10**9
sharded_state_dicts = []
current_block = {}
current_block_size = 0
total_size = 0
for key, weight in checkpoint:
    weight_size = weight.numel() * transformers.modeling_utils.dtype_byte_size(weight.dtype)
    # If this weight is going to tip up over the maximal size, we split.
    if current_block_size + weight_size > max_shard_size:
        sharded_state_dicts.append(current_block)
        current_block = {}
        current_block_size = 0

    current_block[key] = weight
    current_block_size += weight_size
    total_size += weight_size

# Add the last block
sharded_state_dicts.append(current_block)

# If we only have one shard, we return it
if len(sharded_state_dicts) == 1:
    print("saving", os.path.join(destination, "pytorch_model.bin"))
    torch.save(sharded_state_dicts[0], os.path.join(destination, "pytorch_model.bin"))

# Otherwise, let's build the index
else:
    weight_map = {}
    shards = {}
    for idx, shard in enumerate(sharded_state_dicts):
        shard_file = "pytorch_model.bin".replace(".bin", f"-{idx+1:05d}-of-{len(sharded_state_dicts):05d}.bin")
        shards[shard_file] = shard
        for key in shard.keys():
            weight_map[key] = shard_file
        print("saving", os.path.join(destination, shard_file))
        torch.save(shard, os.path.join(destination, shard_file))

    # Add the metadata
    metadata = {"total_size": total_size}
    index = {"metadata": metadata, "weight_map": weight_map}
    print("saving", os.path.join(destination, "pytorch_model.bin.index.json"))
    with open(os.path.join(destination, "pytorch_model.bin.index.json"), "w") as f:
        json.dump(index, f, indent=2)

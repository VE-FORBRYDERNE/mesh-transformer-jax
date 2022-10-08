#!/usr/bin/env python3

import torch
import numpy as np

for i in range(1, 47):
	print(i)
	z = torch.load(f"in/pytorch_model-{i:05}-of-00046.bin", map_location="cpu")
	for k, v in z.items():
		if any(k.endswith(q) for q in (".attention.bias", ".attention.masked_bias", ".attention.rotary_emb.inv_freq")):
			print(i, "(skipped)", k)
			continue
		t = np.asarray(v)
		print(i, t.dtype, k)
		np.save(f"phase1_out/{k}.npy", t, allow_pickle=False)

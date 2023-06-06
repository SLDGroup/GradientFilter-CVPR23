import torch as th
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("path1")
parser.add_argument("path2")

args = parser.parse_args()

ckpt1 = th.load(args.path1, map_location='cpu')
ckpt2 = th.load(args.path2, map_location='cpu')
k1 = set(ckpt1['state_dict'].keys())
k2 = set(ckpt2['state_dict'].keys())
diff = k1.difference(k2)
common = k1.intersection(k2)
print("Key difference:", diff)
for k in sorted(list(common)):
    v1, v2 = ckpt1['state_dict'][k], ckpt2['state_dict'][k]
    if v1.shape != v2.shape:
        print(k, "shape mismatch")
        continue
    abs_diff = (v1 - v2).abs().max()
    if abs_diff > 1e-7:
        print(k, "different value", abs_diff)

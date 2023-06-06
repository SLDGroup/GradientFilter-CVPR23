import numpy as np
import os
from glob import glob
import json
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("pattern", type=str, default='*')
parser.add_argument("output", type=str, default='results.csv')
parser.add_argument("--min-steps", type=int, default=20)
args = parser.parse_args()

log_pattern = os.path.join(args.pattern, "*.log.json")
log_paths = sorted(glob(log_pattern))

results = {}

with open(args.output, 'w') as out:
    for path in log_paths:
        with open(path, 'r') as f:
            seg = path.split('/')
            exp_name = seg[-3]
            version = int(seg[-2].split('_')[-1])
            f.readline()
            max_miou = -1
            max_macc = -1
            max_iter = 0
            for l in f.readlines():
                l = json.loads(l)
                if l.get('mode', 'train') == 'val':
                    if l['mIoU'] > max_miou:
                        max_miou = l['mIoU']
                    if l['mAcc'] > max_macc:
                        max_macc = l['mAcc']
                max_iter = max(max_iter, l.get("iter", 0))
            if max_iter < args.min_steps:
                print(
                    f"Not enough steps: {max_iter}k/{args.min_steps}k\n    {path}")
                continue
            if exp_name not in results:
                results[exp_name] = {}
            results[exp_name][version] = (max_miou, max_macc)
        out.write(f"{exp_name}, {version}, {max_miou}, {max_macc}\n")
    out.write("-"*100)
    out.write("\n")
    for k, v in results.items():
        r = np.array(list(v.values()))
        mean = np.mean(r, axis=0)
        std = np.std(r, axis=0)
        out.write(f"{k:60s}, mean, {mean[0]:.4f}, {mean[1]:.4f}\n")
        out.write(f"{k:60s},  std, {std[0]:.4f}, {std[1]:.4f}\n")

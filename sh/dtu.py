import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('machine', type=str)
args = parser.parse_args()

with open('sh/dir.json') as f:
    d = json.load(f)
d = d[args.machine]

cmd = f"""{d['train_environ']}
    python train.py
    --num_workers {d['num_workers']}
    --data_root {d['dtu_dir']}
    --dataset_name dtu
    --model_name model_cas
    --num_src 3
    --max_d 128
    --interval_scale 1.5
    --cas_depth_num 32,16,8
    --cas_interv_scale 4,2,1
    --resize 800,600
    --crop 640,512
    --mode soft
    --num_samples 160000
    --batch_size {d['batch_size']}
    --job_name temp
    --save_dir {d['save_dir']}
"""

cmd = ' '.join(cmd.strip().split())
print(cmd)
os.system(cmd)
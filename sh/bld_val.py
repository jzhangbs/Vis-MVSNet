import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('machine', type=str)
args = parser.parse_args()

with open('sh/dir.json') as f:
    d = json.load(f)
d = d[args.machine]

for m in ['<job name>']:
    for ns in range(3, 3+1):
        cmd = f"""{d['val_environ']}
            python val.py
            --data_root {d['bld_dir']}
            --dataset_name blended
            --model_name model_cas
            --num_src {ns}
            --max_d 128
            --interval_scale 1
            --cas_depth_num 32,16,8
            --cas_interv_scale 4,2,1
            --resize 768,576
            --crop 768,576
            --mode soft
            --load_path {d['save_dir']}/{m}
        """
        cmd = ' '.join(cmd.strip().split())
        print(cmd)
        os.system(cmd)
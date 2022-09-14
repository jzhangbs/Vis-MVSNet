import argparse
import os
import shutil
import sys
import json
import itertools
sys.path.append('.')
import importlib

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import matplotlib.pyplot as plt
# from apex import amp

from utils.preproc import to_channel_first, resize, random_crop, recursive_apply, image_net_center_inv, scale_camera
from utils.io_utils import load_model, subplot_map, write_cam, write_pfm, write_pair


parser = argparse.ArgumentParser()

parser.add_argument('--data_root', type=str, help='The root dir of the data.')
parser.add_argument('--dataset_name', type=str, default='tanksandtemples', help='The name of the dataset. Should be identical to the dataloader source file. e.g. blended refers to data/blended.py.')
parser.add_argument('--model_name', type=str, default='model_cas', help='The name of the model. Should be identical to the model source file. e.g. model_cas refers to core/model_cas.py.')

parser.add_argument('--num_src', type=int, default=7, help='The number of source views.')
parser.add_argument('--max_d', type=int, default=256, help='The standard max depth number.')
parser.add_argument('--interval_scale', type=float, default=1., help='The standard interval scale.')
parser.add_argument('--cas_depth_num', type=str, default='64,32,16', help='The depth number for each stage.')
parser.add_argument('--cas_interv_scale', type=str, default='4,2,1', help='The interval scale for each stage.')
parser.add_argument('--resize', type=str, default='1920,1080', help='The size of the preprocessed input resized from the original one.')
parser.add_argument('--crop', type=str, default='1920,1056', help='The size of the preprocessed input cropped from the resized one.')

parser.add_argument('--mode', type=str, default='soft', choices=['soft', 'maxpool', 'average', 'aveplus'], help='The fusion strategy.')
parser.add_argument('--occ_guide', action='store_true', default=False, help='Deprecated')

parser.add_argument('--load_path', type=str, default=None, help='The dir of the folder containing the pretrained checkpoints.')
parser.add_argument('--load_step', type=int, default=-1, help='The step to load. -1 for the latest one.')

parser.add_argument('--show_result', action='store_true', default=False, help='Set to show the results.')
parser.add_argument('--write_result', action='store_true', default=False, help='Set to save the results.')
parser.add_argument('--result_dir', type=str, help='The dir to save the results.')

args = parser.parse_args()

if __name__ == '__main__':
    # torch.backends.cudnn.benchmark = True

    [resize_width, resize_height], [crop_width, crop_height] = [[int(v) for v in arg_str.split(',')] for arg_str in [args.resize, args.crop]]
    cas_depth_num = [int(v) for v in args.cas_depth_num.split(',')]
    cas_interv_scale = [float(v) for v in args.cas_interv_scale.split(',')]

    Model = importlib.import_module(f'core.{args.model_name}').Model
    get_val_loader = importlib.import_module(f'data.{args.dataset_name}').get_val_loader

    dataset, loader = get_val_loader(
        args.data_root, args.num_src,
        {
            'interval_scale': args.interval_scale,
            'max_d': args.max_d,
            'resize_width': resize_width,
            'resize_height': resize_height,
            'crop_width': crop_width,
            'crop_height': crop_height
        }
    )

    model = Model()
    model.cuda()
    # model = amp.initialize(model, opt_level='O0')
    model = nn.DataParallel(model)
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters() if p.requires_grad])))

    load_model(model, args.load_path, args.load_step)
    print(f'load {os.path.join(args.load_path, str(args.load_step))}')
    model.eval()

    if args.write_result:
        write_pair(os.path.join(args.result_dir, 'pair.txt'), dataset.pair)

    pbar = tqdm.tqdm(enumerate(loader), dynamic_ncols=True, total=len(loader))
    for i, sample in pbar:
        if sample.get('skip') is not None and np.any(sample['skip']): raise ValueError()
        if sample.get('id') is not None:
            sample_id = str(sample['id'][0])
            del sample['id']
        else:
            sample_id = str(i)

        ref, ref_cam, srcs, srcs_cam, gt, masks = [sample[attr] for attr in ['ref', 'ref_cam', 'srcs', 'srcs_cam', 'gt', 'masks']]
        recursive_apply(sample, lambda x: torch.from_numpy(x).float().cuda())
        ref_t, ref_cam_t, srcs_t, srcs_cam_t, gt_t, masks_t = [sample[attr] for attr in ['ref', 'ref_cam', 'srcs', 'srcs_cam', 'gt', 'masks']]

        with torch.no_grad():
            final_depth, stages = model(sample, cas_depth_num, cas_interv_scale, mode=args.mode, verbose_return=False)
        recursive_apply(stages, lambda x: x.clone().cpu().numpy())

        pbar.set_description(f'{final_depth.shape}')

        if args.show_result or args.write_result:
            prob_maps = [stages[i]['prob_map'][0, 0] for i in range(3)]
            if args.show_result:
                stage_est_depths = [stages[i]['est_depth'][0, 0] for i in range(3)]
                plt_map = [
                    [stage_est_depths[0], stage_est_depths[1], stage_est_depths[2]],
                    [ref[0, 0], srcs[0, 0, 0], srcs[0, 1, 0]],
                    [prob_maps[0], prob_maps[1], prob_maps[2]]
                ]
                subplot_map(plt_map)
                plt.show()
            if args.write_result:
                ref_o = np.transpose(ref[0], [1, 2, 0])
                ref_o = image_net_center_inv(ref_o)
                ref_o = cv2.resize(ref_o, (ref_o.shape[1]//2, ref_o.shape[0]//2), interpolation=cv2.INTER_LINEAR)
                ref_cam_o = ref_cam[0]
                ref_cam_o = scale_camera(ref_cam_o, .5)
                est_depth_o = final_depth[0, 0].detach().cpu().numpy()
                prob_maps_o = prob_maps
                prob_map_combined = np.stack(prob_maps_o, axis=-1)
                cv2.imwrite(os.path.join(args.result_dir, f'{sample_id.zfill(8)}.jpg'), ref_o)
                write_cam(os.path.join(args.result_dir, f'cam_{sample_id.zfill(8)}_flow3.txt'), ref_cam_o)
                write_pfm(os.path.join(args.result_dir, f'{sample_id.zfill(8)}_flow3.pfm'), est_depth_o)
                for stage_i, prob_map_o in enumerate(prob_maps_o):
                    write_pfm(os.path.join(args.result_dir, f'{sample_id.zfill(8)}_flow{stage_i+1}_prob.pfm'), prob_map_o)
                write_pfm(os.path.join(args.result_dir, f'{sample_id.zfill(8)}_prob.pfm'), prob_map_combined)

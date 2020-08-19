import argparse
import os
import sys
import json
import importlib
import itertools
sys.path.append('.')

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import matplotlib.pyplot as plt
# from apex import amp

# from core.model_cas import Model, Loss
from utils.preproc import to_channel_first, resize, random_crop, recursive_apply, image_net_center_inv
# import data.dtu as dtu, data.sceneflow as sf, data.blended as bld
from utils.io_utils import load_model, subplot_map, write_pfm


parser = argparse.ArgumentParser()

parser.add_argument('--data_root', type=str, help='The root dir of the data.')
parser.add_argument('--dataset_name', type=str, default='blended', help='The name of the dataset. Should be identical to the dataloader source file. e.g. blended refers to data/blended.py.')
parser.add_argument('--model_name', type=str, default='model_cas', help='The name of the model. Should be identical to the model source file. e.g. model_cas refers to core/model_cas.py.')

parser.add_argument('--num_src', type=int, default=3, help='The number of source views.')
parser.add_argument('--max_d', type=int, default=128, help='The standard max depth number.')
parser.add_argument('--interval_scale', type=float, default=1., help='The standard interval scale.')
parser.add_argument('--cas_depth_num', type=str, default='32,16,8', help='The depth number for each stage.')
parser.add_argument('--cas_interv_scale', type=str, default='4,2,1', help='The interval scale for each stage.')
parser.add_argument('--resize', type=str, default='768,576', help='The size of the preprocessed input resized from the original one.')
parser.add_argument('--crop', type=str, default='768,576', help='The size of the preprocessed input cropped from the resized one.')

parser.add_argument('--mode', type=str, default='soft', choices=['soft', 'hard', 'uwta', 'maxpool', 'average'], help='The fusion strategy.')
parser.add_argument('--occ_guide', action='store_true', default=False, help='Deprecated')

parser.add_argument('--load_path', type=str, default=None, help='The dir of the folder containing the pretrained checkpoints.')
parser.add_argument('--load_step', type=int, default=-1, help='The step to load. -1 for the latest one.')

parser.add_argument('--show_result', action='store_true', default=False, help='Set to show the results.')
parser.add_argument('--write_result', action='store_true', default=False, help='Set to save the results.')
parser.add_argument('--result_dir', type=str, help='The dir to save the results.')

args = parser.parse_args()

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True

    [resize_width, resize_height], [crop_width, crop_height] = [[int(v) for v in arg_str.split(',')] for arg_str in [args.resize, args.crop]]
    cas_depth_num = [int(v) for v in args.cas_depth_num.split(',')]
    cas_interv_scale = [float(v) for v in args.cas_interv_scale.split(',')]

    Model = importlib.import_module(f'core.{args.model_name}').Model
    Loss = importlib.import_module(f'core.{args.model_name}').Loss
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
    compute_loss = Loss()

    load_model(model, args.load_path, args.load_step)
    print(f'load {os.path.join(args.load_path, str(args.load_step))}')
    model.eval()

    loss_history = []
    less1_history = []
    less3_history = []

    pbar = tqdm.tqdm(enumerate(loader), dynamic_ncols=True, total=len(loader))
    # pbar = itertools.product(range(num_scan), range(num_ref), range(num_view))
    for i, sample in pbar:
        if sample.get('skip') is not None and np.any(sample['skip']): continue

        ref, ref_cam, srcs, srcs_cam, gt, masks = [sample[attr] for attr in ['ref', 'ref_cam', 'srcs', 'srcs_cam', 'gt', 'masks']]
        recursive_apply(sample, lambda x: torch.from_numpy(x).float().cuda())
        ref_t, ref_cam_t, srcs_t, srcs_cam_t, gt_t, masks_t = [sample[attr] for attr in ['ref', 'ref_cam', 'srcs', 'srcs_cam', 'gt', 'masks']]

        with torch.no_grad():
            # est_depth, prob_map, pair_results = model([ref_t, ref_cam_t, srcs_t, srcs_cam_t], args.max_d, mode=args.mode)  #MVS
            outputs, refined_depth, prob_maps = model(sample, cas_depth_num, cas_interv_scale, mode=args.mode)
            [[est_depth_1, pair_results_1], [est_depth_2, pair_results_2], [est_depth, pair_results]] = outputs
            # est_depth = model([ref_t, ref_cam_t, srcs_t, srcs_cam_t, gt_t], args.max_d)
            # losses = compute_loss([est_depth, pair_results], gt_t, masks_t, ref_cam_t, args.max_d, occ_guide=args.occ_guide, mode=args.mode)  #MVS
            losses = compute_loss([outputs, refined_depth], gt_t, masks_t, ref_cam_t, args.max_d, occ_guide=args.occ_guide, mode=args.mode)
            # losses = compute_loss(est_depth, gt_t, masks_t, ref_cam_t, args.max_d, strict_mask=False)
        losses_np = [v.item() for v in losses[:5]]  #MVS
        loss, uncert_loss, less1, less3, l1 = losses_np  #MVS
        # loss, less1, less3, l1 = losses_np

        est_depth, prob_map = [arr.clone().cpu().data.numpy() for arr in [est_depth, prob_maps[2]]]
        # est_depth, prob_map = [arr.clone().cpu().data.numpy() for arr in [est_depth, prob_map]]
        recursive_apply(pair_results, lambda x: x.clone().cpu().data.numpy())  #MVS

        stats = losses[5]
        stats_np = [(l1.item(), less1.item(), less3.item()) for l1, less1, less3 in stats]
        stats_str = ''.join([f'({l1:.3f} {less1*100:.2f} {less3*100:.2f})' for l1, less1, less3 in stats_np])

        # pbar.set_description(f'loss: {loss:.3f} uncert_loss: {uncert_loss:.3f} less1: {less1:.4f} less3: {less3:.4f} l1: {l1:.3f}')  #MVS
        pbar.set_description(f'{loss:.3f}{stats_str}{l1:.3f}')
        # pbar.set_description(f'loss: {loss:.3f} less1: {less1:.4f} less3: {less3:.4f} l1: {l1:.3f}')
        if not np.isnan(l1):
            loss_history.append(l1)
            less1_history.append(less1)
            less3_history.append(less3)

        if (i % 49 == 0 or True) and (args.show_result or args.write_result):
            abs_err_scaled, in_range = [v.clone().cpu().data.numpy() for v in losses[-2:]]  #MVS
            fused_uncert = -np.log(sum([np.exp(-uncert[0, 0, ...]) for _, (uncert, occ) in pair_results]))
            fused_prob_lb = np.clip(1-2*np.exp(2*fused_uncert)/(4*3/args.num_src)**2, 0, 1)
            ref_rgb = image_net_center_inv(np.transpose(ref[0, ...], [1, 2, 0]))[..., ::-1]
            srcs_rgb = [image_net_center_inv(np.transpose(srcs[0, i, ...], [1, 2, 0]))[..., ::-1] for i in range(3)]
            error_map = abs_err_scaled * in_range
            error_map = np.clip(error_map, 0, 50)
            error_map = (error_map - np.min(error_map)) / (np.max(error_map) - np.min(error_map))
            error_map = error_map ** 0.5
            plt_map = [
                [est_depth[0, 0, ...], gt[0, 0, ...], error_map[0, 0, ...], fused_uncert],
                [ref_rgb, srcs_rgb[0], srcs_rgb[1], srcs_rgb[2]],
                [prob_map[0, 0, ...], pair_results[0][0][0, 0, ...], pair_results[1][0][0, 0, ...], pair_results[2][0][0, 0, ...]],  #MVS
                [fused_prob_lb, pair_results[0][1][0][0, 0, ...], pair_results[1][1][0][0, 0, ...], pair_results[2][1][0][0, 0, ...]],
            ]
            subplot_map(plt_map)
            plt.gcf().set_size_inches(23, 13)
            # print(est_depth[0, 0, ...])
            if args.write_result:
                # write_pfm(os.path.join(args.result_dir, f'uncert_{i}.pfm'), pair_results[0][1][0][0, 0, ...])
                plt.savefig(fname=os.path.join(args.result_dir, f'fig{i:03}_{l1:.3f}.png'))
            if args.show_result:
                plt.show()
            plt.clf()

        del pair_results, est_depth, losses

    avg_loss, avg_less1, avg_less3 = [sum(arr)/len(arr) for arr in [loss_history, less1_history, less3_history]]
    print(f'avg l1: {avg_loss:.3f} less1: {avg_less1:.4f} less3: {avg_less3:.4f}')

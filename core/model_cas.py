import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Callable, Any
import numpy as np
import itertools
import time
import matplotlib.pyplot as plt

from core.nn_utils import ListModule, UNet, multi_dims, CSPN, soft_argmin, entropy, StackCGRU, hourglass, bin_op_reduce, groupwise_correlation, normalize_for_grid_sample, get_in_range
from utils.preproc import scale_camera, recursive_apply
from core.homography import get_pixel_grids, get_homographies, homography_warping
from utils.utils import NanError

cpg = 8
ZERO = torch.tensor(0).float().cuda()

class FeatExt(nn.Module):

    def __init__(self):
        super(FeatExt, self).__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2, 2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.unet = UNet(16, 2, 1, 2, [], [32, 64, 128], [], '2d', 2)
        self.final_conv_1 = nn.Conv2d(128, 32, 3, 1, 1, bias=False)
        self.final_conv_2 = nn.Conv2d(64, 32, 3, 1, 1, bias=False)
        self.final_conv_3 = nn.Conv2d(32, 32, 3, 1, 1, bias=False)

    def forward(self, x):
        out = self.init_conv(x)
        out1, out2, out3 = self.unet(out, multi_scale=3)
        return self.final_conv_1(out1), self.final_conv_2(out2), self.final_conv_3(out3)


class Reg(nn.Module):

    def __init__(self):
        super(Reg, self).__init__()
        self.init_conv = lambda x: x
        self.unet = UNet(8, 1, 0, 4, [], [8, 16], [], 'reg1', dim=3)

    def forward(self, x):
        init = self.init_conv(x)
        out = self.unet(init)
        return out


class RegPair(nn.Module):

    def __init__(self):
        super(RegPair, self).__init__()
        self.final_conv = nn.Conv3d(8, 1, 3, 1, 1, bias=False)

    def forward(self, x):
        out = self.final_conv(x)
        return out


class RegFuse(nn.Module):

    def __init__(self):
        super(RegFuse, self).__init__()
        self.init_conv = lambda x: x
        self.unet = UNet(8, 1, 0, 4, [], [8, 16], [], 'reg2', dim=3)
        self.final_conv = nn.Conv3d(8, 1, 3, 1, 1, bias=False)

    def forward(self, x):
        init = self.init_conv(x)
        out = self.unet(init)
        out = self.final_conv(out)
        return out


class UncertNet(nn.Module):

    def __init__(self):
        super(UncertNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        self.head_conv = nn.Conv2d(8, 1, 3, 1, 1, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + x
        out = self.head_conv(out)
        return out


class SingleStage(nn.Module):

    def __init__(self):
        super(SingleStage, self).__init__()
        # self.feat_ext = FeatExt()
        self.reg = Reg()
        self.reg_fuse = RegFuse()
        self.reg_pair = RegPair()  #MVS
        self.uncert_net = UncertNet()  #MVS

    def build_cost_volume(self, ref, ref_cam, src, src_cam, depth_num, depth_start, depth_interval, s_scale, d_scale):
        ref_cam_scaled, src_cam_scaled = [scale_camera(cam, 1 / s_scale) for cam in [ref_cam, src_cam]]
        Hs = get_homographies(ref_cam_scaled, src_cam_scaled, depth_num//d_scale, depth_start, depth_interval*d_scale)
        # ndhw33
        src_nd_c_h_w = src.unsqueeze(1).repeat(1, depth_num//d_scale, 1, 1, 1).reshape(-1, *src.shape[1:])  # n*d chw
        warped_src_nd_c_h_w = homography_warping(src_nd_c_h_w, Hs.reshape(-1, *Hs.shape[2:]))  # n*d chw
        warped_src = warped_src_nd_c_h_w.reshape(-1, depth_num//d_scale, *src.shape[1:]).transpose(1, 2)  # ncdhw
        return warped_src

    def build_cost_volume_2(self, ref, ref_cam, src, src_cam, depth_num, depth_start, depth_interval, s_scale, d_scale):
        with torch.no_grad():
            ref_cam, src_cam = [scale_camera(cam, 1 / s_scale) for cam in [ref_cam, src_cam]]

            Kr = ref_cam[:,1:2,:,:]  # n144
            Kr[:,:,3,:3] = 0
            Kr[:,:,3,3] = 1
            Rr = ref_cam[:,0:1,:,:]  # n144
            Ks = src_cam[:,1:2,:,:]  # n144
            Ks[:,:,3,:3] = 0
            Ks[:,:,3,3] = 1
            Rs = src_cam[:,0:1,:,:]  # n144

            n, _, h, w = ref.shape
            d = depth_num = depth_num//d_scale
            depth_interval = depth_interval*d_scale

            depth = depth_start + depth_interval * torch.arange(0, depth_num, dtype=ref.dtype, device=ref.device).reshape(1,d,1,1)  # nd11/ndhw
            uv = 0.5 + torch.from_numpy(np.mgrid[:ref.shape[2], :ref.shape[3]]).to(ref.dtype).to(ref.device)  # 2hw
            uv = uv.flip(0).permute(1,2,0)[None, None, :,:,:, None].repeat(n,d,1,1,1,1)  # ndhw21
            depth_inv = (torch.ones(1,1,h,w, dtype=ref.dtype, device=ref.device) / depth)[..., None, None]  # ndhw11
            uv = torch.cat([uv, torch.ones_like(uv[...,-1:,:]), depth_inv], dim=-2)  # ndhw41 (u,v,1,1/d)
            uv_flat = uv.reshape(n, -1, 4, 1)  # nN41

            uv_src_flat = Ks @ Rs @ Rr.inverse() @ Kr.inverse() @ uv_flat  # nN41
            uv_src_flat = uv_src_flat[:,:,:2,0] / uv_src_flat[:,:,2:3,0]  # nN2
            fetch_uv = normalize_for_grid_sample(src, uv_src_flat.unsqueeze(1))  # n1N2
            warped_mask = get_in_range(fetch_uv).reshape(n,1,d,h,w)  # n1dhw

        warped_src = F.grid_sample(src, fetch_uv, mode='bilinear', padding_mode='zeros', align_corners=False)  # nc1N
        warped_src = warped_src.reshape(n,-1,d,h,w)  # ncdhw

        return warped_src, warped_mask

    def forward(self, sample, depth_num, depth_start, depth_interval, mode='soft', s_scale=1, verbose_return=True):
        #                                n111/n1hw    n111
        ref_feat, ref_cam, srcs_feat, srcs_cam = sample

        ref_ncdhw = ref_feat.unsqueeze(2).repeat(1, 1, depth_num, 1, 1)
        pair_results = []  #MVS

        if mode == 'soft':
            weight_sum = torch.zeros((ref_ncdhw.shape[0], 1, 1, *ref_ncdhw.shape[3:])).to(ref_ncdhw.dtype).cuda()
        if mode == 'average':
            pass
        if mode == 'aveplus':
            weight_sum = torch.zeros((ref_ncdhw.shape[0], 1, *ref_ncdhw.shape[2:])).to(ref_ncdhw.dtype).cuda()
        if mode == 'maxpool':
            init = True
        fused_interm = torch.zeros((ref_ncdhw.shape[0], 8, *ref_ncdhw.shape[2:])).to(ref_ncdhw.dtype).cuda()

        for src_feat, src_cam in zip(srcs_feat, srcs_cam):
            warped_src, warped_mask = self.build_cost_volume_2(ref_feat, ref_cam, src_feat, src_cam, depth_num, depth_start, depth_interval, s_scale, 1)
            cost_volume = groupwise_correlation(ref_ncdhw, warped_src, 8, 1)
            interm = self.reg(cost_volume)
            if verbose_return:
                score_volume = self.reg_pair(interm)  # n1dhw
                score_volume = score_volume.squeeze(1)  # ndhw
                prob_volume, est_depth_class = soft_argmin(score_volume, dim=1, keepdim=True)
                est_depth = est_depth_class * depth_interval + depth_start
                if mode in ['soft']:
                    entropy_ = entropy(prob_volume, dim=1, keepdim=True)
                    uncert = self.uncert_net(entropy_)
                else:
                    uncert = None
            else:
                est_depth = uncert = prob_volume = None
            pair_results.append({
                'est_depth': est_depth,
                'uncert': uncert,
                'prob_vol': prob_volume,
            })

            if mode == 'soft':
                weight = (-uncert).exp().unsqueeze(2)
                weight_sum = weight_sum + weight
                fused_interm = fused_interm + interm * weight
            if mode == 'average':
                weight = None
                fused_interm = fused_interm + interm
            if mode == 'aveplus':
                weight = warped_mask
                weight_sum = weight_sum + weight
                fused_interm = fused_interm + interm * weight
            if mode == 'maxpool':
                weight = None
                if init:
                    fused_interm = fused_interm + interm
                    init = False
                else:
                    fused_interm = torch.max(fused_interm, interm)

        if mode == 'soft':
            fused_interm = fused_interm / weight_sum
        if mode == 'average':
            fused_interm = fused_interm / len(srcs_feat)
        if mode == 'aveplus':
            fused_interm = fused_interm / (weight_sum + 1e-9)
        if mode == 'maxpool':
            pass

        score_volume = self.reg_fuse(fused_interm)  # n1dhw
        score_volume = score_volume.squeeze(1)  # ndhw

        prob_volume, est_depth_class, prob_map = soft_argmin(score_volume, dim=1, keepdim=True, window=2)
        est_depth = est_depth_class * depth_interval + depth_start

        return est_depth, prob_map, pair_results, (prob_volume if verbose_return else None)


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.feat_ext = FeatExt()
        self.stage1 = SingleStage()
        self.stage2 = SingleStage()
        self.stage3 = SingleStage()
    
    def forward(self, sample, depth_nums, interval_scales, mode='soft', verbose_return=True):
        ref, ref_cam, srcs, srcs_cam = [sample[attr] for attr in ['ref', 'ref_cam', 'srcs', 'srcs_cam']]
        depth_start = ref_cam[:,1,3,0].reshape(-1,1,1,1)  # n111
        depth_interval = ref_cam[:,1,3,1].reshape(-1,1,1,1)  # n111
        srcs_cam = [srcs_cam[:, i, ...] for i in range(srcs_cam.shape[1])]

        n, v, c, h, w = srcs.shape
        img_pack = torch.cat([ref, srcs.transpose(0, 1).reshape(v*n, c, h, w)])
        feat_pack_1, feat_pack_2, feat_pack_3 = self.feat_ext(img_pack)

        ref_feat_1, *srcs_feat_1 = [feat_pack_1[i*n:(i+1)*n] for i in range(v+1)]
        est_depth_1, prob_map_1, pair_results_1, prob_vol_1 = self.stage1(
            [ref_feat_1, ref_cam, srcs_feat_1, srcs_cam],
            depth_num=depth_nums[0], depth_start=depth_start,
            depth_interval=depth_interval*interval_scales[0],
            mode=mode, s_scale=8, verbose_return=verbose_return
        )
        prob_map_1_up = F.interpolate(prob_map_1, scale_factor=4, mode='bilinear', align_corners=False)

        ref_feat_2, *srcs_feat_2 = [feat_pack_2[i*n:(i+1)*n] for i in range(v+1)]
        depth_start_2 = \
            F.interpolate(est_depth_1.detach(), size=(ref_feat_2.shape[2], ref_feat_2.shape[3]), mode='bilinear', align_corners=False) \
            - depth_nums[1] * depth_interval * interval_scales[1] / 2
        est_depth_2, prob_map_2, pair_results_2, prob_vol_2 = self.stage2(
            [ref_feat_2, ref_cam, srcs_feat_2, srcs_cam],
            depth_num=depth_nums[1], depth_start=depth_start_2,
            depth_interval=depth_interval*interval_scales[1],
            mode=mode, s_scale=4, verbose_return=verbose_return
        )
        prob_map_2_up = F.interpolate(prob_map_2, scale_factor=2, mode='bilinear', align_corners=False)

        ref_feat_3, *srcs_feat_3 = [feat_pack_3[i*n:(i+1)*n] for i in range(v+1)]
        depth_start_3 = \
            F.interpolate(est_depth_2.detach(), size=(ref_feat_3.shape[2], ref_feat_3.shape[3]), mode='bilinear', align_corners=False) \
            - depth_nums[2] * depth_interval * interval_scales[2] / 2
        est_depth_3, prob_map_3, pair_results_3, prob_vol_3 = self.stage3(
            [ref_feat_3, ref_cam, srcs_feat_3, srcs_cam],
            depth_num=depth_nums[2], depth_start=depth_start_3,
            depth_interval=depth_interval*interval_scales[2],
            mode=mode, s_scale=2, verbose_return=verbose_return
        )

        final_depth = est_depth_3

        stages = [
            {
                'est_depth': est_depth_1,
                'prob_map': prob_map_1_up,
            },
            {
                'est_depth': est_depth_2,
                'prob_map': prob_map_2_up,
            },
            {
                'est_depth': est_depth_3,
                'prob_map': prob_map_3,
            },
        ]
        if verbose_return:
            stages[0].update({
                'depth_start': depth_start,
                'depth_interval': depth_interval*interval_scales[0],
                'depth_num': torch.tensor(depth_nums[0], dtype=torch.long, device=depth_start.device).repeat(n).reshape(-1,1,1,1),
                'pair_results': pair_results_1,
                'prob_vol': prob_vol_1,
            })
            stages[1].update({
                'depth_start': depth_start_2,
                'depth_interval': depth_interval*interval_scales[1],
                'depth_num': torch.tensor(depth_nums[1], dtype=torch.long, device=depth_start.device).repeat(n).reshape(-1,1,1,1),
                'pair_results': pair_results_2,
                'prob_vol': prob_vol_2,
            })
            stages[2].update({
                'depth_start': depth_start_3,
                'depth_interval': depth_interval*interval_scales[2],
                'depth_num': torch.tensor(depth_nums[2], dtype=torch.long, device=depth_start.device).repeat(n).reshape(-1,1,1,1),
                'pair_results': pair_results_3,
                'prob_vol': prob_vol_3,
            })
        return final_depth, stages


class Loss(nn.Module):  # TODO

    def __init__(self):
        super(Loss, self).__init__()
        # self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def label2prob(self, d_class, d_num):
        d_num = d_num[0,0,0,0].item()
        d_vol = torch.arange(d_num, dtype=d_class.dtype, device=d_class.device).reshape(1,-1,1,1)  # 1d11
        gt_prob_vol = (1 - (d_class - d_vol).abs()).clamp(0,1)
        return gt_prob_vol

    def forward(self, outputs, gt, masks, ref_cam, max_d, mode='soft'):
        final_depth, stages = outputs

        depth_start = ref_cam[:, 1:2, 3:4, 0:1]  # n111
        depth_interval = ref_cam[:, 1:2, 3:4, 1:2]  # n111
        depth_end = depth_start + (max_d - 1) * depth_interval
        # masks = [masks[:, i, ...] for i in range(masks.shape[1])]

        stage_losses = []
        stats = []
        for s in stages:
            cas_depth_start = s['depth_start']
            cas_depth_interval = s['depth_interval']
            cas_depth_num = s['depth_num']
            cas_depth_end = cas_depth_start + (cas_depth_num - 1) * cas_depth_interval

            est_depth = s['est_depth']
            gt_downsized = F.interpolate(gt, size=(est_depth.shape[2], est_depth.shape[3]), mode='bilinear', align_corners=False)
            # gt_class = (gt_downsized - cas_depth_start) / cas_depth_interval
            # gt_prob_vol = self.label2prob(gt_class, cas_depth_num)
            valid = (gt_downsized >= depth_start) & (gt_downsized <= depth_end)
            # valid_cas = (gt_downsized >= cas_depth_start) & (gt_downsized <= cas_depth_end)

            # ===== stage l1 =====
            abs_err = (est_depth - gt_downsized).abs()
            abs_err_scaled = abs_err / depth_interval
            # filtered_l1 = abs_err_scaled[valid_cas].mean()
            # filtered_ce = self.cross_entropy(s['prob_vol'], gt_prob_vol).unsqueeze(1)[valid_cas].mean()

            # ===== pair l1 =====
            pair_results = s['pair_results']
            pair_abs_err = [(p['est_depth'] - gt_downsized).abs() for p in pair_results]
            pair_abs_err_scaled = [err / depth_interval for err in pair_abs_err]
            # pair_ce = [self.cross_entropy(p['prob_vol'], gt_prob_vol).unsqueeze(1) for p in pair_results]
            pair_l1_losses = [loss_map[valid].mean() for loss_map in pair_abs_err_scaled]
            pair_l1_loss = sum(pair_l1_losses) / len(pair_l1_losses)

            # ===== uncert =====
            if mode in ['soft']:
                uncert_losses = [
                    (err[valid] * (-p['uncert'][valid]).exp() + p['uncert'][valid]).mean()
                    for err, p in zip(pair_abs_err_scaled, pair_results)
                ]
                uncert_loss = sum(uncert_losses) / len(uncert_losses)

            # ===== for display =====
            l1 = abs_err_scaled[valid].mean()
            less1 = (abs_err_scaled[valid] < 1.).to(gt.dtype).mean()
            less3 = (abs_err_scaled[valid] < 3.).to(gt.dtype).mean()

            pair_loss = pair_l1_loss
            if mode in ['soft']:
                pair_loss = pair_loss + uncert_loss
            loss = l1 + pair_loss
            stage_losses.append(loss)
            stats.append((l1, less1, less3))
        
        abs_err = (final_depth - gt_downsized).abs()
        abs_err_scaled = abs_err / depth_interval
        l1 = abs_err_scaled[valid].mean()
        less1 = (abs_err_scaled[valid] < 1.).to(gt.dtype).mean()
        less3 = (abs_err_scaled[valid] < 3.).to(gt.dtype).mean()
        
        loss = stage_losses[0]*0.5 + stage_losses[1]*1.0 + stage_losses[2]*2.0

        return loss, pair_loss, less1, less3, l1, stats, abs_err_scaled, valid

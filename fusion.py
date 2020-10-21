import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import os

from data.depth2pcd import get_val_loader
from core.homography import get_pixel_grids
from core.nn_utils import bin_op_reduce
from utils.io_utils import subplot_map

parser = argparse.ArgumentParser()

parser.add_argument('--data', type=str, default='')
parser.add_argument('--pair', type=str, default='')
parser.add_argument('--view', type=int, default=10)
parser.add_argument('--vthresh', type=int, default=4)
parser.add_argument('--pthresh', type=str, default='.8,.7,.8')
parser.add_argument('--cam_scale', type=float, default=1)
parser.add_argument('--show_result', action='store_true', default=False)

args = parser.parse_args()


def idx_img2cam(idx_img_homo, depth, cam):  # nhw31, n1hw -> nhw41
    idx_cam = cam[:,1:2,:3,:3].unsqueeze(1).inverse() @ idx_img_homo  # nhw31
    idx_cam = idx_cam / (idx_cam[...,-1:,:]+1e-9) * depth.permute(0,2,3,1).unsqueeze(4)  # nhw31
    idx_cam_homo = torch.cat([idx_cam, torch.ones_like(idx_cam[...,-1:,:])], dim=-2)  # nhw41
    # FIXME: out-of-range is 0,0,0,1, will have valid coordinate in world
    return idx_cam_homo


def idx_cam2world(idx_cam_homo, cam):  # nhw41 -> nhw41
    idx_world_homo =  cam[:,0:1,...].unsqueeze(1).inverse() @ idx_cam_homo  # nhw41
    idx_world_homo = idx_world_homo / (idx_world_homo[...,-1:,:]+1e-9)  # nhw41
    return idx_world_homo


def idx_world2cam(idx_world_homo, cam):  # nhw41 -> nhw41
    idx_cam_homo = cam[:,0:1,...].unsqueeze(1) @ idx_world_homo  # nhw41
    idx_cam_homo = idx_cam_homo / (idx_cam_homo[...,-1:,:]+1e-9)   # nhw41
    return idx_cam_homo


def idx_cam2img(idx_cam_homo, cam):  # nhw41 -> nhw31
    idx_cam = idx_cam_homo[...,:3,:] / (idx_cam_homo[...,3:4,:]+1e-9)  # nhw31
    idx_img_homo = cam[:,1:2,:3,:3].unsqueeze(1) @ idx_cam  # nhw31
    idx_img_homo = idx_img_homo / (idx_img_homo[...,-1:,:]+1e-9)
    return idx_img_homo


def project_img(src_img, dst_depth, src_cam, dst_cam, height=None, width=None):  # nchw, n1hw -> nchw, n1hw
    if height is None: height = src_img.size()[-2]
    if width is None: width = src_img.size()[-1]
    dst_idx_img_homo = get_pixel_grids(height, width).unsqueeze(0)  # nhw31
    dst_idx_cam_homo = idx_img2cam(dst_idx_img_homo, dst_depth, dst_cam)  # nhw41
    dst_idx_world_homo = idx_cam2world(dst_idx_cam_homo, dst_cam)  # nhw41
    dst2src_idx_cam_homo = idx_world2cam(dst_idx_world_homo, src_cam)  # nhw41
    dst2src_idx_img_homo = idx_cam2img(dst2src_idx_cam_homo, src_cam)  # nhw31
    warp_coord = dst2src_idx_img_homo[...,:2,0]  # nhw2
    warp_coord[..., 0] /= width
    warp_coord[..., 1] /= height
    warp_coord = (warp_coord*2-1).clamp(-1.1, 1.1)  # nhw2
    in_range = bin_op_reduce([-1<=warp_coord[...,0], warp_coord[...,0]<=1, -1<=warp_coord[...,1], warp_coord[...,1]<=1], torch.min).to(src_img.dtype).unsqueeze(1)  # n1hw
    warped_img = F.grid_sample(src_img, warp_coord, mode='bilinear', padding_mode='zeros', align_corners=False)
    return warped_img, in_range


def prob_filter(ref_probs, pthresh, greater=True):  # n31hw -> n1hw
    cmpr = lambda x, y: x > y if greater else lambda x, y: x < y
    masks = cmpr(ref_probs, torch.Tensor(pthresh).to(ref_probs.dtype).to(ref_probs.device).view(1,3,1,1,1)).to(ref_probs.dtype)
    mask = (masks.sum(dim=1) >= (len(pthresh)-0.1))
    return mask


def get_reproj(ref_depth, srcs_depth, ref_cam, srcs_cam):  # n1hw, nv1hw -> n1hw
    n, v, _, h, w = srcs_depth.size()
    srcs_depth_f = srcs_depth.view(n*v, 1, h, w)
    srcs_cam_f = srcs_cam.view(n*v, 2, 4, 4)
    ref_depth_r = ref_depth.unsqueeze(1).repeat(1,v,1,1,1).view(n*v, 1, h, w)
    ref_cam_r = ref_cam.unsqueeze(1).repeat(1,v,1,1,1).view(n*v, 2, 4, 4)
    idx_img = get_pixel_grids(h, w).unsqueeze(0)  # 1hw31

    srcs_idx_cam = idx_img2cam(idx_img, srcs_depth_f, srcs_cam_f)  # Nhw41
    srcs_idx_world = idx_cam2world(srcs_idx_cam, srcs_cam_f)  # Nhw41
    srcs2ref_idx_cam = idx_world2cam(srcs_idx_world, ref_cam_r)  # Nhw41
    srcs2ref_idx_img = idx_cam2img(srcs2ref_idx_cam, ref_cam_r)  # Nhw31
    srcs2ref_xyd = torch.cat([srcs2ref_idx_img[...,:2,0], srcs2ref_idx_cam[...,2:3,0]], dim=-1).permute(0,3,1,2)  # N3hw

    reproj_xyd_f, in_range_f= project_img(srcs2ref_xyd, ref_depth_r, srcs_cam_f, ref_cam_r)  # N3hw, N1hw
    reproj_xyd = reproj_xyd_f.view(n,v,3,h,w)
    in_range = in_range_f.view(n,v,1,h,w)
    return reproj_xyd, in_range


def vis_filter(ref_depth, reproj_xyd, in_range, img_dist_thresh, depth_thresh, vthresh):
    n, v, _, h, w = reproj_xyd.size()
    xy = get_pixel_grids(h, w).permute(3,2,0,1).unsqueeze(1)[:,:,:2]  # 112hw
    dist_masks = (reproj_xyd[:,:,:2,:,:] - xy).norm(dim=2, keepdim=True) < img_dist_thresh  # nv1hw
    depth_masks = (ref_depth.unsqueeze(1) - reproj_xyd[:,:,2:,:,:]).abs() < (torch.max(ref_depth.unsqueeze(1), reproj_xyd[:,:,2:,:,:])*depth_thresh)  # nv1hw
    masks = bin_op_reduce([in_range, dist_masks.to(ref_depth.dtype), depth_masks.to(ref_depth.dtype)], torch.min)  # nv1hw
    mask = masks.sum(dim=1) >= (vthresh-1.1)  # n1hw
    return masks, mask


def ave_fusion(ref_depth, reproj_xyd, masks):
    ave = ((reproj_xyd[:,:,2:,:,:]*masks).sum(dim=1)+ref_depth) / (masks.sum(dim=1)+1)  # n1hw
    return ave


if __name__ == '__main__':
    
    dataset, loader = get_val_loader(args.data, args.pair, args.view, {})
    pthresh = [float(v) for v in args.pthresh.split(',')]

    views = {}

    pbar = tqdm.tqdm(loader, dynamic_ncols=True)
    for sample_np in pbar:
        if sample_np.get('skip') is not None and np.any(sample_np['skip']): continue
        sample = {attr: torch.from_numpy(sample_np[attr]).float().cuda() for attr in sample_np if attr not in ['skip', 'id']}

        prob_mask = prob_filter(sample['ref_probs'], pthresh)

        reproj_xyd, in_range = get_reproj(*[sample[attr] for attr in ['ref_depth', 'srcs_depth', 'ref_cam', 'srcs_cam']])
        vis_masks, vis_mask = vis_filter(sample['ref_depth'], reproj_xyd, in_range, 1, 0.01, args.vthresh)

        ref_depth_ave = ave_fusion(sample['ref_depth'], reproj_xyd, vis_masks)

        mask = bin_op_reduce([prob_mask, vis_mask], torch.min)

        if args.show_result:
            subplot_map([
                [sample['ref_depth'][0,0].cpu().data.numpy(), ref_depth_ave[0,0].cpu().data.numpy(), (ref_depth_ave*mask)[0,0].cpu().data.numpy()],
                [prob_mask[0,0].cpu().data.numpy(), vis_mask[0,0].cpu().data.numpy(), mask[0,0].cpu().data.numpy()]
            ])
            plt.show()
        
        idx_img = get_pixel_grids(*ref_depth_ave.size()[-2:]).unsqueeze(0)
        idx_cam = idx_img2cam(idx_img, ref_depth_ave, sample['ref_cam'])
        points = idx_cam2world(idx_cam, sample['ref_cam'])[...,:3,0].permute(0,3,1,2)
        points_np = points.cpu().data.numpy()
        mask_np = mask.cpu().data.numpy()
        for i in range(points_np.shape[0]):
            p_f_list = [points_np[i,k][mask_np[i,0]] for k in range(3)]
            p_f = np.stack(p_f_list, -1)
            c_f_list = [sample_np['ref'][i,k][mask_np[i,0]] for k in range(3)]
            c_f = np.stack(c_f_list, -1) / 255
            ref_id = str(sample_np['id'][i])
            views[ref_id] = (p_f, c_f)
    
    print('Write combined PCD')
    p_all, c_all = [np.concatenate([v[k] for key, v in views.items()], axis=0) for k in range(2)]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(p_all)
    pcd.colors = o3d.utility.Vector3dVector(c_all)
    o3d.io.write_point_cloud(os.path.join(args.data, f'all_torch.ply'), pcd)

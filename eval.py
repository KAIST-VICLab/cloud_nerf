import os
import cv2

from collections import defaultdict
from tqdm import tqdm
import imageio
from argparse import ArgumentParser

from models.rendering import render_rays
from models.nerf import *
from models.cloud_code import *

from utils import load_ckpt
import metrics
import lpips

from datasets import dataset_dict
from datasets.depth_utils import *

# colmap
from datasets.colmap_utils import read_cameras_binary, read_images_binary, read_points3d_binary
from datasets.ray_utils import *
from datasets.llff import center_poses

from train import NeRFSystem

torch.backends.cudnn.benchmark = True


def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        default='/home/ubuntu/data/nerf_example_data/nerf_synthetic/lego',
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='blender',
                        choices=['blender', 'llff'],
                        help='which dataset to validate')
    parser.add_argument('--scene_name', type=str, default='test',
                        help='scene name, used as output folder name')
    parser.add_argument('--split', type=str, default='test',
                        help='test or test_train')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[800, 800],
                        help='resolution (img_w, img_h) of the image')
    parser.add_argument('--spheric_poses', default=False, action="store_true",
                        help='whether images are taken in spheric poses (for llff)')

    parser.add_argument('--N_emb_xyz', type=int, default=10,
                        help='number of frequencies in xyz positional encoding')
    parser.add_argument('--N_emb_dir', type=int, default=4,
                        help='number of frequencies in dir positional encoding')
    parser.add_argument('--N_samples', type=int, default=64,
                        help='number of coarse samples')
    parser.add_argument('--N_importance', type=int, default=128,
                        help='number of additional fine samples')
    parser.add_argument('--use_disp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument('--chunk', type=int, default=32 * 1024 * 4,
                        help='chunk size to split the input to avoid OOM')

    parser.add_argument('--weight_path', type=str, required=True,
                        help='pretrained checkpoint path to load')

    parser.add_argument('--save_depth', default=False, action="store_true",
                        help='whether to save depth prediction')
    parser.add_argument('--depth_format', type=str, default='pfm',
                        choices=['pfm', 'bytes'],
                        help='which format to save')

    return parser.parse_args()


@torch.no_grad()
def batched_inference(models, embeddings,
                      rays, N_samples, N_importance, use_disp,
                      chunk):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    results = defaultdict(list)
    for i in range(0, B, chunk):
        rendered_ray_chunks = \
            render_rays(models,
                        embeddings,
                        rays[i:i + chunk],
                        N_samples,
                        use_disp,
                        0,
                        0,
                        N_importance,
                        chunk,
                        dataset.white_back,
                        test_time=True)

        for k, v in rendered_ray_chunks.items():
            results[k] += [v.cpu()]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results


def read_colmap_meta(hparams):
    # Step 1: rescale focal length according to training resolution
    camdata = read_cameras_binary(os.path.join(
        hparams.root_dir, 'sparse/0/cameras.bin'))
    H = camdata[1].height
    W = camdata[1].width
    focal = camdata[1].params[0] * hparams.img_wh[0] / W

    # Step 2: correct poses
    # read extrinsics (of successfully reconstructed images)
    imdata = read_images_binary(os.path.join(
        hparams.root_dir, 'sparse/0/images.bin'))

    w2c_mats = []
    bottom = np.array([0, 0, 0, 1.]).reshape(1, 4)
    for k in imdata:
        im = imdata[k]
        R = im.qvec2rotmat()
        t = im.tvec.reshape(3, 1)
        w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
    w2c_mats = np.stack(w2c_mats, 0)
    # (N_images, 3, 4) cam2world matrices
    poses = np.linalg.inv(w2c_mats)[:, :3]

    pts3d = read_points3d_binary(os.path.join(
        hparams.root_dir, 'sparse/0/points3D.bin'))
    pts_world = np.zeros((1, 3, len(pts3d)))  # (1, 3, N_points)
    bounds = np.zeros((len(poses), 2))  # (N_images, 2)
    visibilities = np.zeros((len(poses), len(pts3d))
                            )  # (N_images, N_points)

    for i, k in enumerate(pts3d):
        pts_world[0, :, i] = pts3d[k].xyz
        for j in pts3d[k].image_ids:
            visibilities[j - 1, i] = 1
    depths = ((pts_world - poses[..., 3:4]) * poses[..., 2:3]).sum(1)
    for i in range(len(poses)):
        visibility_i = visibilities[i]
        zs = depths[i][visibility_i == 1]
        bounds[i] = [np.percentile(zs, 0.1), np.percentile(zs, 99.9)]
        valid_depth = (depths[i] >= bounds[i][0]) & (
            depths[i] <= bounds[i][1])
        visibility_i = visibility_i.astype(bool) & valid_depth
        visibilities[i] = visibility_i.astype(np.float64)

    # ! IMPORTANT REMOVE OUTLIERS BY USING PER IMAGE BOUND
    valid_points = np.any(visibilities, axis=0)
    pts_world = np.transpose(pts_world[0])[valid_points]  # (N_points, 3)

    # fps
    fps_kps = fps(
        pts_world, config['code_cloud']['num_codes'])

    # ! IMPORTANT WE NEED TO TRANSLATE WORLD COORDS TO AVG POSE ORIGIN
    poses = np.concatenate(
        [poses[..., 0:1], -poses[..., 1:3], poses[..., 3:4]], -1)
    poses, pose_avg = center_poses(poses)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg

    pts_world_homo = np.concatenate(
        [pts_world, np.ones((pts_world.shape[0], 1))], axis=1)
    fps_kps_homo = np.concatenate(
        [fps_kps, np.ones((fps_kps.shape[0], 1))], axis=1)

    trans_pts_world = np.linalg.inv(
        pose_avg_homo) @ pts_world_homo[:, :, None]
    trans_fps_kps = np.linalg.inv(pose_avg_homo) @ fps_kps_homo[:, :, None]

    # ! scale the nearest points after cenetering
    kps = torch.from_numpy(trans_pts_world[:, :3, 0])
    fps_kps = torch.from_numpy(trans_fps_kps[:, :3, 0])
    near_original = bounds.min()
    scale_factor = near_original * 0.75  # 0.75 is the default parameter
    kps /= scale_factor
    fps_kps /= scale_factor

    # np.save('assets/code_centered.npy', kps)
    # breakpoint()

    # convert to ndc
    kps_ndc = get_ndc_coor(
        hparams.img_wh[1], hparams.img_wh[0], focal, 1.0, kps)
    # np.save('assets/code_ndc.npy', kps_ndc)
    # breakpoint()
    fps_kps_ndc = get_ndc_coor(hparams.img_wh[1], hparams.img_wh[0],
                               focal, 1.0, fps_kps)
    # np.save('assets/fps_code_ndc.npy', fps_kps_ndc)
    # breakpoint()

    return kps_ndc, fps_kps_ndc


if __name__ == "__main__":
    args = get_opts()
    w, h = args.img_wh
    lpips_vgg = lpips.LPIPS(net='vgg').cuda()
    lpips_vgg = lpips_vgg.eval()

    kwargs = {'root_dir': args.root_dir,
              'split': args.split,
              'img_wh': tuple(args.img_wh),
              'val_num': 3}
    if args.dataset_name == 'llff':
        kwargs['spheric_poses'] = args.spheric_poses
    dataset = dataset_dict[args.dataset_name](**kwargs)

    system = NeRFSystem(args)

    # embedding_xyz = Embedding(args.N_emb_xyz)
    # embedding_dir = Embedding(args.N_emb_dir)

    # kps, fps_kps = read_colmap_meta(args)

    # if args.N_importance == 0:
    #     nerf_coarse = CloudNeRF(kps, fps_kps, 6 * args.N_emb_xyz + 3, 6 *
    #                             args.N_emb_dir + 3)
    #     nerf_coarse.cuda().eval()
    #     load_ckpt(nerf_coarse, args.ckpt_path, 'nerf_coarse')
    #     models = {'coarse': nerf_coarse}

    # else:
    #     nerf_fine = CloudNeRF(kps, fps_kps, 6 * args.N_emb_xyz + 3, 6 *
    #                           args.N_emb_dir + 3)
    #     nerf_fine.cuda().eval()

    #     load_ckpt(nerf_fine, args.ckpt_path, 'nerf_fine')
    #     models = {'fine': nerf_fine}

    embedding_xyz = system.embedding_xyz
    embedding_dir = system.embedding_dir
    models = system.models
    for k in models.keys():
        models[k] = models[k].cuda().eval()

    imgs, depth_maps, psnrs, mean_lpips, ssims = [], [], [], [], []
    dir_name = f'results/{args.dataset_name}/{args.scene_name}'
    os.makedirs(dir_name, exist_ok=True)

    embeddings = {'xyz': embedding_xyz, 'dir': embedding_dir}

    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        rays = sample['rays'].cuda()
        results = batched_inference(models, embeddings, rays,
                                    args.N_samples, args.N_importance, args.use_disp,
                                    args.chunk)
        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        img_pred = np.clip(results[f'rgb_{typ}'].view(
            h, w, 3).cpu().numpy(), 0, 1)

        if args.save_depth:
            depth_pred = results[f'depth_{typ}'].view(h, w).cpu().numpy()
            depth_maps += [depth_pred]
            if args.depth_format == 'pfm':
                save_pfm(os.path.join(
                    dir_name, f'depth_{i:03d}.pfm'), depth_pred)
            else:
                with open(os.path.join(dir_name, f'depth_{i:03d}'), 'wb') as f:
                    f.write(depth_pred.tobytes())

        img_pred_ = (img_pred * 255).astype(np.uint8)
        imgs += [img_pred_]
        imageio.imwrite(os.path.join(dir_name, f'{i:03d}.png'), img_pred_)

        img_gt_ = (sample['rgbs'].view(
            h, w, 3).cpu().numpy() * 255).astype(np.uint8)
        imageio.imwrite(os.path.join(dir_name, f'gt_{i:03d}.png'), img_gt_)

        if 'rgbs' in sample:
            rgbs = sample['rgbs']
            img_gt = rgbs.view(h, w, 3)

            # scale to compute lpips
            scaled_gt = img_gt * 2.0 - 1.0
            scaled_pred = img_pred * 2.0 - 1.0
            scaled_pred = torch.from_numpy(scaled_pred)
            lpips_val = lpips_vgg(scaled_gt[:, :, [2, 1, 0]].permute(2, 0, 1).unsqueeze(
                0).cuda(), scaled_pred[:, :, [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).cuda())
            mean_lpips.append(lpips_val.detach().squeeze().cpu().numpy())
            psnrs += [metrics.psnr(img_gt, img_pred).item()]

            # compute ssim
            ssim = metrics.ssim(img_gt.permute(2, 0, 1)[None], torch.from_numpy(
                img_pred).permute(2, 0, 1)[None])
            ssims.append(ssim)

    imageio.mimsave(os.path.join(
        dir_name, f'{args.scene_name}.gif'), imgs, fps=30)

    if args.save_depth:
        min_depth = np.min(depth_maps)
        max_depth = np.max(depth_maps)
        depth_imgs = (depth_maps - np.min(depth_maps)) / \
            (max(np.max(depth_maps) - np.min(depth_maps), 1e-8))
        depth_imgs_ = [cv2.applyColorMap(
            (img * 255).astype(np.uint8), cv2.COLORMAP_JET) for img in depth_imgs]
        imageio.mimsave(os.path.join(
            dir_name, f'{args.scene_name}_depth.gif'), depth_imgs_, fps=30)

    if psnrs:
        mean_psnr = np.mean(psnrs)
        print(psnrs)
        print(f'Mean PSNR : {mean_psnr:.3f}')

    if ssims:
        mean_ssim = np.mean(ssims)
        print(ssims)
        print(f'Mean SSIM : {mean_ssim:.3f}')

    if mean_lpips:
        print(mean_lpips)
        mean_lpips = np.mean(np.array(mean_lpips))
        print(f'Mean LPIPS : {mean_lpips:.3f}')

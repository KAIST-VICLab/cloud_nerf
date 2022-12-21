import os
from argparse import ArgumentParser
from collections import defaultdict

import cv2
import imageio
import lpips
import metrics
import numpy as np
import torch
from datasets import dataset_dict
from datasets.depth_utils import save_pfm
from models.rendering import render_rays
from tqdm import tqdm
from train import NeRFSystem


torch.backends.cudnn.benchmark = True


def get_opts():
    parser = ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/home/ubuntu/data/nerf_example_data/nerf_synthetic/lego",
        help="root directory of dataset",
    )
    parser.add_argument(
        "--dataset_name", type=str, default="blender", choices=["blender", "llff"], help="which dataset to validate"
    )
    parser.add_argument("--scene_name", type=str, default="test",
                        help="scene name, used as output folder name")
    parser.add_argument("--split", type=str, default="test",
                        help="test or test_train")
    parser.add_argument(
        "--img_wh", nargs="+", type=int, default=[800, 800], help="resolution (img_w, img_h) of the image"
    )
    parser.add_argument(
        "--spheric_poses",
        default=False,
        action="store_true",
        help="whether images are taken in spheric poses (for llff)",
    )

    parser.add_argument("--N_emb_xyz", type=int, default=10,
                        help="number of frequencies in xyz positional encoding")
    parser.add_argument("--N_emb_dir", type=int, default=4,
                        help="number of frequencies in dir positional encoding")
    parser.add_argument("--N_samples", type=int, default=64,
                        help="number of coarse samples")
    parser.add_argument("--N_importance", type=int, default=128,
                        help="number of additional fine samples")
    parser.add_argument("--use_disp", default=False,
                        action="store_true", help="use disparity depth sampling")
    parser.add_argument("--chunk", type=int, default=32 * 1024 * 4,
                        help="chunk size to split the input to avoid OOM")

    parser.add_argument("--weight_path", type=str, required=True,
                        help="pretrained checkpoint path to load")

    parser.add_argument("--save_depth", default=False,
                        action="store_true", help="whether to save depth prediction")
    parser.add_argument(
        "--depth_format", type=str, default="pfm", choices=["pfm", "bytes"], help="which format to save"
    )

    return parser.parse_args()


@torch.no_grad()
def batched_inference(models, embeddings, rays, N_samples, N_importance, use_disp, chunk):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    results = defaultdict(list)
    for i in range(0, B, chunk):
        rendered_ray_chunks = render_rays(
            models,
            embeddings,
            rays[i: i + chunk],
            N_samples,
            use_disp,
            0,
            0,
            N_importance,
            chunk,
            dataset.white_back,
            test_time=True,
        )

        for k, v in rendered_ray_chunks.items():
            results[k] += [v.cpu()]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results


if __name__ == "__main__":
    args = get_opts()
    w, h = args.img_wh
    lpips_vgg = lpips.LPIPS(net="vgg").cuda()
    lpips_vgg = lpips_vgg.eval()

    kwargs = {"root_dir": args.root_dir, "split": args.split,
              "img_wh": tuple(args.img_wh), "val_num": 3}
    if args.dataset_name == "llff":
        kwargs["spheric_poses"] = args.spheric_poses
    dataset = dataset_dict[args.dataset_name](**kwargs)

    system = NeRFSystem(args)

    embedding_xyz = system.embedding_xyz
    embedding_dir = system.embedding_dir
    models = system.models
    for k in models.keys():
        models[k] = models[k].cuda().eval()

    imgs, depth_maps, psnrs, mean_lpips, ssims = [], [], [], [], []
    dir_name = f"results/{args.dataset_name}/{args.scene_name}"
    os.makedirs(dir_name, exist_ok=True)

    embeddings = {"xyz": embedding_xyz, "dir": embedding_dir}

    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        rays = sample["rays"].cuda()
        results = batched_inference(
            models, embeddings, rays, args.N_samples, args.N_importance, args.use_disp, args.chunk
        )
        typ = "fine" if "rgb_fine" in results else "coarse"

        img_pred = np.clip(results[f"rgb_{typ}"].view(
            h, w, 3).cpu().numpy(), 0, 1)

        if args.save_depth:
            depth_pred = results[f"depth_{typ}"].view(h, w).cpu().numpy()
            depth_maps += [depth_pred]
            if args.depth_format == "pfm":
                save_pfm(os.path.join(
                    dir_name, f"depth_{i:03d}.pfm"), depth_pred)
            else:
                with open(os.path.join(dir_name, f"depth_{i:03d}"), "wb") as f:
                    f.write(depth_pred.tobytes())

        img_pred_ = (img_pred * 255).astype(np.uint8)
        imgs += [img_pred_]
        imageio.imwrite(os.path.join(dir_name, f"{i:03d}.png"), img_pred_)

        img_gt_ = (sample["rgbs"].view(
            h, w, 3).cpu().numpy() * 255).astype(np.uint8)
        imageio.imwrite(os.path.join(dir_name, f"gt_{i:03d}.png"), img_gt_)

        if "rgbs" in sample:
            rgbs = sample["rgbs"]
            img_gt = rgbs.view(h, w, 3)

            # scale to compute lpips
            scaled_gt = img_gt * 2.0 - 1.0
            scaled_pred = img_pred * 2.0 - 1.0
            scaled_pred = torch.from_numpy(scaled_pred)
            lpips_val = lpips_vgg(
                scaled_gt[:, :, [2, 1, 0]].permute(
                    2, 0, 1).unsqueeze(0).cuda(),
                scaled_pred[:, :, [2, 1, 0]].permute(
                    2, 0, 1).unsqueeze(0).cuda(),
            )
            mean_lpips.append(lpips_val.detach().squeeze().cpu().numpy())
            psnrs += [metrics.psnr(img_gt, img_pred).item()]

            # compute ssim
            ssim = metrics.ssim(img_gt.permute(2, 0, 1)[None], torch.from_numpy(
                img_pred).permute(2, 0, 1)[None])
            ssims.append(ssim)

    imageio.mimsave(os.path.join(
        dir_name, f"{args.scene_name}.gif"), imgs, fps=30)

    if args.save_depth:
        min_depth = np.min(depth_maps)
        max_depth = np.max(depth_maps)
        depth_imgs = (depth_maps - np.min(depth_maps)) / \
            (max(np.max(depth_maps) - np.min(depth_maps), 1e-8))
        depth_imgs_ = [cv2.applyColorMap(
            (img * 255).astype(np.uint8), cv2.COLORMAP_JET) for img in depth_imgs]
        imageio.mimsave(os.path.join(
            dir_name, f"{args.scene_name}_depth.gif"), depth_imgs_, fps=30)

    if psnrs:
        mean_psnr = np.mean(psnrs)
        print(psnrs)
        print(f"Mean PSNR : {mean_psnr:.3f}")

    if ssims:
        mean_ssim = np.mean(ssims)
        print(ssims)
        print(f"Mean SSIM : {mean_ssim:.3f}")

    if mean_lpips:
        print(mean_lpips)
        mean_lpips = np.mean(np.array(mean_lpips))
        print(f"Mean LPIPS : {mean_lpips:.3f}")

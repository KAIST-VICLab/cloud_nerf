import glob
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from datasets import dataset_dict

# colmap
from datasets.colmap_utils import read_cameras_binary, read_images_binary, read_points3d_binary
from datasets.llff import center_poses
from datasets.ray_utils import get_ndc_coor, read_gen

# losses
from losses import loss_dict

# metrics
from metrics import psnr
from models.cloud_code import CloudNeRF, config

# models
from models.nerf import Embedding
from models.rendering import render_rays
from opt import get_opts

# fps
from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation

# pytorch-lightning
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
from torch.utils.data import DataLoader

# optimizer, scheduler, visualization
from utils import get_learning_rate, get_optimizer, get_scheduler, load_ckpt, visualize_depth


@torch.no_grad()
def fps(points, n_samples):
    points = torch.from_numpy(points).unsqueeze(
        0).float().cuda()  # 1, N_points, 3
    points_flipped = points.transpose(1, 2).contiguous()
    fps_index = furthest_point_sample(points, n_samples)  # 1, n_samples

    fps_kps = gather_operation(points_flipped, fps_index).transpose(
        1, 2).contiguous().squeeze(0)  # n_samples, 3
    return fps_kps.cpu().numpy()


class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.root_dir = hparams.root_dir

        self.loss = loss_dict["color"](coef=1)

        self.embedding_xyz = Embedding(hparams.N_emb_xyz)
        self.embedding_dir = Embedding(hparams.N_emb_dir)
        self.embeddings = {"xyz": self.embedding_xyz,
                           "dir": self.embedding_dir}

        kps, fps_kps = self.read_colmap_meta(hparams)

        if hparams.N_importance == 0:
            self.nerf_coarse = CloudNeRF(
                kps, fps_kps, 6 * hparams.N_emb_xyz + 3, 6 * hparams.N_emb_dir + 3)

            self.models = {"coarse": self.nerf_coarse}
            load_ckpt(self.nerf_coarse, hparams.weight_path, "nerf_coarse")

        else:
            self.nerf_fine = CloudNeRF(
                kps, fps_kps, 6 * hparams.N_emb_xyz + 3, 6 * hparams.N_emb_dir + 3)

            self.models = {"fine": self.nerf_fine}
            load_ckpt(self.nerf_fine, hparams.weight_path, "nerf_fine")

    def read_colmap_meta(self, hparams):
        camdata = read_cameras_binary(os.path.join(
            hparams.root_dir, "sparse/0/cameras.bin"))
        self.origin_intrinsics = camdata
        W = camdata[1].width
        self.focal = camdata[1].params[0] * hparams.img_wh[0] / W

        imdata = read_images_binary(os.path.join(
            hparams.root_dir, "sparse/0/images.bin"))

        w2c_mats = []
        bottom = np.array([0, 0, 0, 1.0]).reshape(1, 4)
        for k in imdata:
            im = imdata[k]
            R = im.qvec2rotmat()
            t = im.tvec.reshape(3, 1)
            w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
        w2c_mats = np.stack(w2c_mats, 0)
        # (N_images, 3, 4) cam2world matrices
        poses = np.linalg.inv(w2c_mats)[:, :3]
        self.origin_extrinsics = poses

        pts3d = read_points3d_binary(os.path.join(
            hparams.root_dir, "sparse/0/points3D.bin"))

        mvs_points = self.load_mvs_depth().numpy()
        near_bound = mvs_points.min(axis=0)[-1]
        pts3d = {k: v for (k, v) in pts3d.items() if v.xyz[-1] > near_bound}

        pts_world = np.zeros((1, 3, len(pts3d)))  # (1, 3, N_points)
        self.bounds = np.zeros((len(poses), 2))  # (N_images, 2)
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
            self.bounds[i] = [np.percentile(zs, 0.1), np.percentile(zs, 99.9)]
            valid_depth = (depths[i] >= self.bounds[i][0]) & (
                depths[i] <= self.bounds[i][1])
            visibility_i = visibility_i.astype(bool) & valid_depth
            visibilities[i] = visibility_i.astype(np.float64)

        valid_points = np.any(visibilities, axis=0)
        pts_world = np.transpose(pts_world[0])[valid_points]  # (N_points, 3)

        # fps
        fps_kps = fps(pts_world, config["code_cloud"]["num_codes"])
        global_kps = fps(mvs_points, pts_world.shape[0])

        pts_world = np.concatenate([pts_world, global_kps], axis=0)
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

        kps = torch.from_numpy(trans_pts_world[:, :3, 0])
        fps_kps = torch.from_numpy(trans_fps_kps[:, :3, 0])
        near_original = self.bounds.min()
        scale_factor = near_original * 0.75  # 0.75 is the default parameter
        kps /= scale_factor
        fps_kps /= scale_factor

        # convert to ndc
        kps_ndc = get_ndc_coor(
            hparams.img_wh[1], hparams.img_wh[0], self.focal, 1.0, kps)
        fps_kps_ndc = get_ndc_coor(
            hparams.img_wh[1], hparams.img_wh[0], self.focal, 1.0, fps_kps)
        return kps_ndc, fps_kps_ndc

    def load_mvs_depth(self):
        depth_glob = os.path.join(self.root_dir, "depths", "*.pfm")
        self.depth_list = sorted(glob.glob(depth_glob))
        depths = []
        for i in range(len(self.depth_list)):
            depth = read_gen(self.depth_list[i])

            depths.append(depth)
        self.depths = np.stack(depths, 0).astype(np.float32)  # N x H x W

        per_view_points = self.project_to_3d()
        mvs_points = self.fwd_consistency_check(per_view_points)
        return mvs_points

    def project_to_3d(self):
        N, H, W = self.depths.shape
        focal = self.origin_intrinsics[1].params[0]
        origin_h, origin_w = self.origin_intrinsics[1].height, self.origin_intrinsics[1].width

        origin_cy, origin_cx = self.origin_intrinsics[1].params[2], self.origin_intrinsics[1].params[1]

        origin_K = np.array([[focal, 0, origin_cx, 0], [
                            0, focal, origin_cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        origin_K[0, :] /= origin_w
        origin_K[1, :] /= origin_h

        self.normalized_K = origin_K

        mvs_K = self.normalized_K.copy()
        mvs_K[0, :] *= W
        mvs_K[1, :] *= H
        self.mvs_K = mvs_K

        inv_mvs_K = np.linalg.pinv(mvs_K)
        inv_mvs_K = torch.from_numpy(inv_mvs_K)

        # create mesh grid for mvs image
        meshgrid = np.meshgrid(range(W), range(H), indexing="xy")
        id_coords = (np.stack(meshgrid, axis=0).astype(
            np.float32)).reshape(2, -1)
        id_coords = torch.from_numpy(id_coords)

        ones = torch.ones(N, 1, H * W)

        pix_coords = torch.unsqueeze(torch.stack(
            [id_coords[0].view(-1), id_coords[1].view(-1)], 0), 0)
        pix_coords = pix_coords.repeat(N, 1, 1)
        pix_coords = torch.cat([pix_coords, ones], 1)

        # project to cam coord
        inv_mvs_K = inv_mvs_K[None, ...].repeat(N, 1, 1).float()
        cam_points = torch.matmul(inv_mvs_K[:, :3, :3], pix_coords)
        mvs_depth = torch.from_numpy(
            self.depths).float().unsqueeze(1).view(N, 1, -1)
        cam_points = mvs_depth * cam_points
        cam_points = torch.cat([cam_points, ones], 1)

        # project to world coord
        T = torch.from_numpy(self.origin_extrinsics).float()
        world_points = torch.matmul(T, cam_points)
        world_points = world_points.permute(0, 2, 1)  # N, H*W, 3

        return world_points

    def fwd_consistency_check(self, per_view_points):
        N, H, W = self.depths.shape
        global_valid_points = []
        for view_id in range(per_view_points.shape[0]):
            curr_view_points = per_view_points[view_id].transpose(
                1, 0)  # 3, H*W
            homo_view_points = torch.cat(
                [curr_view_points, torch.ones(1, H * W)], dim=0)  # 4, H*W
            homo_view_points = homo_view_points.unsqueeze(
                0).repeat(N, 1, 1)  # N,4,H*W

            # project to camera space
            T = torch.from_numpy(self.origin_extrinsics).float()
            homo_T = torch.cat([T, torch.zeros(N, 1, 4)], dim=1)
            homo_T[:, -1, -1] = 1
            inv_T = torch.inverse(homo_T)
            cam_points = torch.matmul(inv_T[:, :3, :], homo_view_points)

            # project to image space
            mvs_K = torch.from_numpy(self.mvs_K).unsqueeze(
                0).repeat(N, 1, 1).float()
            cam_points = torch.matmul(mvs_K[:, :3, :3], cam_points)
            cam_points[:, :2, :] /= cam_points[:, 2:, :]

            z_values = cam_points[:, 2:, :].view(N, 1, H, W)  # N,1,H,W
            xy_coords = cam_points[:, :2, :].transpose(
                2, 1).view(N, H, W, 2)  # N,H,W,2

            xy_coords[..., 0] /= W - 1
            xy_coords[..., 1] /= H - 1
            xy_coords = (xy_coords - 0.5) * 2

            mvs_depth = torch.from_numpy(
                self.depths).float().unsqueeze(1)  # N,1,H,W
            ref_z_values = F.grid_sample(
                mvs_depth, xy_coords, mode="bilinear", align_corners=False)
            err = z_values - 0.9 * ref_z_values

            visible_mask = ref_z_values != 0
            visible_count = visible_mask.int().sum(0)
            valid_visible = visible_count >= 1
            valid_points = err >= 0
            valid_points = torch.all(
                valid_points, dim=0) & valid_visible  # 1,H,W
            global_valid_points.append(valid_points)
        global_valid_points = torch.cat(
            global_valid_points, dim=0).view(N, H * W)  # N,H,W

        filtered_points = per_view_points[global_valid_points, :]
        return filtered_points

    def forward(self, rays):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            rendered_ray_chunks = render_rays(
                self.models,
                self.embeddings,
                rays[i: i + self.hparams.chunk],
                self.hparams.N_samples,
                self.hparams.use_disp,
                self.hparams.perturb,
                self.hparams.noise_std,
                self.hparams.N_importance,
                self.hparams.chunk,  # chunk size is effective in val mode
                self.train_dataset.white_back,
            )

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {"root_dir": self.hparams.root_dir,
                  "img_wh": tuple(self.hparams.img_wh)}
        if self.hparams.dataset_name == "llff":
            kwargs["val_num"] = 3
        self.train_dataset = dataset(split="train", **kwargs)
        self.val_dataset = dataset(split="val", **kwargs)

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models)
        scheduler = get_scheduler(self.hparams, self.optimizer)
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, shuffle=True, num_workers=4, batch_size=self.hparams.batch_size, pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=4,
            # validate one image (H*W rays) at a time
            batch_size=1,
            pin_memory=True,
        )

    def training_step(self, batch, batch_nb):
        rays, rgbs = batch["rays"], batch["rgbs"]
        results = self(rays)
        loss = self.loss(results, rgbs)

        with torch.no_grad():
            typ = "fine" if "rgb_fine" in results else "coarse"
            psnr_ = psnr(results[f"rgb_{typ}"], rgbs)

        self.log("lr", get_learning_rate(self.optimizer))
        self.log("train/loss", loss)
        self.log("train/psnr", psnr_, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_nb):
        rays, rgbs = batch["rays"], batch["rgbs"]
        rays = rays.squeeze()  # (H*W, 3)
        rgbs = rgbs.squeeze()  # (H*W, 3)
        results = self(rays)
        log = {"val_loss": self.loss(results, rgbs)}
        typ = "fine" if "rgb_fine" in results else "coarse"

        if batch_nb == 0:
            W, H = self.hparams.img_wh
            img = results[f"rgb_{typ}"].view(
                H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
            depth = visualize_depth(
                results[f"depth_{typ}"].view(H, W))  # (3, H, W)
            stack = torch.stack([img_gt, img, depth])  # (3, 3, H, W)
            self.logger.experiment.add_images(
                "val/GT_pred_depth", stack, self.global_step)

        psnr_ = psnr(results[f"rgb_{typ}"], rgbs)
        log["val_psnr"] = psnr_

        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        mean_psnr = torch.stack([x["val_psnr"] for x in outputs]).mean()

        self.log("val/loss", mean_loss)
        self.log("val/psnr", mean_psnr, prog_bar=True)


def main(hparams):
    system = NeRFSystem(hparams)
    ckpt_cb = ModelCheckpoint(
        dirpath=f"ckpts/{hparams.exp_name}", filename="{epoch:d}", monitor="val/psnr", mode="max", save_top_k=5
    )
    pbar = TQDMProgressBar(refresh_rate=1)
    callbacks = [ckpt_cb, pbar]

    logger = TensorBoardLogger(
        save_dir="logs", name=hparams.exp_name, default_hp_metric=False)

    trainer = Trainer(
        max_epochs=hparams.num_epochs,
        callbacks=callbacks,
        resume_from_checkpoint=hparams.ckpt_path,
        logger=logger,
        enable_model_summary=False,
        accelerator="auto",
        devices=hparams.num_gpus,
        num_sanity_val_steps=1,
        benchmark=True,
        profiler="simple" if hparams.num_gpus == 1 else None,
        strategy=DDPPlugin(
            find_unused_parameters=False) if hparams.num_gpus > 1 else None,
    )

    trainer.fit(system)


if __name__ == "__main__":
    hparams = get_opts()
    main(hparams)

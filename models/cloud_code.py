from datetime import datetime

import faiss  # make faiss available
import faiss.contrib.torch_utils
import torch
import torch.nn as nn


# ! CloudNeRF configs
config = {}
config["code_cloud"] = {}
config["code_cloud"]["num_codes"] = 8192  # 8192
config["code_cloud"]["num_neighbors"] = 32  # 32

config["code_cloud"]["code_dim"] = 64  # 64
config["code_cloud"]["dist_scale"] = 3.0
config["code_regularization_lambda"] = 0.0
config["code_position_lambda"] = 0
config["distortion_lambda"] = 0


@torch.no_grad()
def find_knn(gpu_index, locs, current_codes, neighbor=config["code_cloud"]["num_neighbors"]):
    n_points = locs.shape[0]
    # Search with torch GPU using pre-allocated arrays
    new_d_square_torch_gpu = torch.zeros(
        n_points, neighbor, device=locs.device, dtype=torch.float32)
    new_i_torch_gpu = torch.zeros(
        n_points, neighbor, device=locs.device, dtype=torch.int64)

    # update current codes
    gpu_index.add(current_codes)

    gpu_index.search(locs, neighbor, new_d_square_torch_gpu, new_i_torch_gpu)
    gpu_index.reset()

    return new_d_square_torch_gpu, new_i_torch_gpu


class CodeCloud(nn.Module):
    def __init__(self, config, num_records, keypoints, fps_keypoints):
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Building CodeCloud.")
        super().__init__()
        self.config = config
        self.SH_basis_dim = 9
        self.origin_keypoints = nn.Parameter(
            torch.Tensor(keypoints.float())[None, ...].repeat(num_records, 1, 1), requires_grad=False
        )

        self.codes_position = nn.Parameter(torch.Tensor(fps_keypoints.float())[
                                           None, ...].repeat(num_records, 1, 1))
        self.codes = nn.Parameter(torch.randn(
            num_records, config["num_codes"], config["code_dim"]) * 0.01)

        self.knn = self.init_knn()

        num_params = sum(p.data.nelement() for p in self.parameters())
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
              "CodeCloud done(#parameters=%d)." % num_params)

    def init_knn(self):
        faiss_cfg = faiss.GpuIndexFlatConfig()
        faiss_cfg.useFloat16 = True
        faiss_cfg.device = 0

        return faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), 3, faiss_cfg)

    def query(self, indices, query_points):
        """
        Args:
            indices: tensor, (batch_size,)
            query_points: tensor, (batch_size, num_points, 3)
        Returns:
            query_codes: tensor, (batch_size, num_points, code_dim)
            square_dist: tensor, (batch_size, num_points, num_codes)
            weight: tensor, (batch_size, num_points, num_codes)
        """
        batch_codes_position = self.codes_position[indices]

        # ! NO GRAD: Need to recompute the square distance for gradients
        _, new_i_torch_gpu = find_knn(
            self.knn, query_points[0], batch_codes_position[0])

        square_dist = (query_points.unsqueeze(2) - self.codes_position[indices][:, new_i_torch_gpu, :]).pow(2).sum(
            dim=-1
        ) + 1e-16

        weight = 1.0 / (torch.sqrt(square_dist) ** self.config["dist_scale"])
        weight = weight / weight.sum(dim=-1, keepdim=True)

        query_codes = torch.matmul(weight[0].unsqueeze(
            1), self.codes[indices][:, new_i_torch_gpu, :][0]).squeeze(1)

        return query_codes

    def get_proposal(self, pts):
        n_rays, n_samples, _ = pts.shape
        new_d_torch_gpu, _ = find_knn(
            self.knn, pts.flatten(0, 1), self.origin_keypoints[0])

        farthest_d = new_d_torch_gpu[:, -1].view(n_rays, n_samples)

        farthest_d[farthest_d > 0.2] = 0.2

        weight = 1.0 / (torch.sqrt(farthest_d) + 1e-10)
        weight = weight / weight.sum(dim=-1, keepdim=True)
        return weight


class IM_Decoder(nn.Module):
    def __init__(self, D=8, W=256, in_channels_xyz=63, in_channels_dir=27, skips=[4]):
        print(datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"), "Building IM-decoder.")
        super(IM_Decoder, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.skips = skips

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W + in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
            nn.Linear(W + in_channels_dir, W // 2), nn.ReLU(True))

        # output layers
        self.sigma = nn.Linear(W, 1)
        self.rgb = nn.Linear(W // 2, 3)

        num_params = sum(p.data.nelement() for p in self.parameters())
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
              "IM decoder done(#parameters=%d)." % num_params)

    def forward(self, x):
        input_xyz, input_dir = torch.split(
            x, [self.in_channels_xyz, self.in_channels_dir], dim=-1)
        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        sigma = self.sigma(xyz_)

        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        rgb = self.rgb(dir_encoding)

        out = torch.cat([rgb, sigma], -1)

        return out


class CloudNeRF(nn.Module):
    def __init__(self, keypoints, fps_kps, input_ch, input_ch_views, num_records=1):
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Building network.")
        super().__init__()
        global config
        self.config = config

        self.code_cloud = CodeCloud(
            config["code_cloud"], num_records, keypoints, fps_kps)
        self.decoder = IM_Decoder(
            D=4,
            W=128,
            in_channels_xyz=config["code_cloud"]["code_dim"] + input_ch,
            in_channels_dir=input_ch_views,
            skips=[2],
        )

        num_params = sum(p.data.nelement() for p in self.parameters())
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
              "Network done(#parameters=%d)." % num_params)

    def forward(self, indices, query_points, xyzdir_embedded):
        """
        Args:
            indices: tensor, (batch_size,)
            query_points: tensor, (batch_size, num_points, 3)
        Returns:
            pred_sd: tensor, (batch_size, num_points)
        """
        query_codes = self.code_cloud.query(indices, query_points[None, ...])

        batch_input = torch.cat([query_codes, xyzdir_embedded], dim=-1)
        pred_sd = self.decoder(batch_input)

        return pred_sd


################################## SH FEATURES ##############################################
class SHCodeCloud(nn.Module):
    def __init__(self, config, num_records, keypoints, fps_keypoints):
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Building CodeCloud.')
        super().__init__()
        self.config = config
        self.SH_basis_dim = 9
        self.origin_keypoints = nn.Parameter(torch.Tensor(
            keypoints.float())[None, ...].repeat(num_records, 1, 1), requires_grad=False)

        self.codes_position = nn.Parameter(torch.Tensor(
            fps_keypoints.float())[None, ...].repeat(num_records, 1, 1))
        self.codes = nn.Parameter(torch.randn(
            num_records, config['num_codes'], config['code_dim']) * 0.01)
        self.sh_codes = nn.Parameter(torch.randn(
            num_records, config['num_codes'], config['code_dim'] * self.SH_basis_dim) * 0.01)

        self.knn = self.init_knn()

        num_params = sum(p.data.nelement() for p in self.parameters())
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
              'CodeCloud done(#parameters=%d).' % num_params)

    def init_knn(self):
        faiss_cfg = faiss.GpuIndexFlatConfig()
        faiss_cfg.useFloat16 = True
        faiss_cfg.device = 0

        return faiss.GpuIndexFlatL2(
            faiss.StandardGpuResources(), 3, faiss_cfg)

    def query(self, indices, query_points, viewdirs):
        """
        Args:
            indices: tensor, (batch_size,)
            query_points: tensor, (batch_size, num_points, 3)
        Returns:
            query_codes: tensor, (batch_size, num_points, code_dim)
            square_dist: tensor, (batch_size, num_points, num_codes)
            weight: tensor, (batch_size, num_points, num_codes)
        """
        batch_codes_position = self.codes_position[indices]

        # ! NO GRAD: Need to recompute the square distance for gradients
        _, new_i_torch_gpu = find_knn(
            self.knn, query_points[0], batch_codes_position[0])

        sh_feat = self.sh_codes[indices][:, new_i_torch_gpu, :][0]
        sh_feat = sh_feat.reshape(-1, self.config['num_neighbors'],
                                  self.config['code_dim'], self.SH_basis_dim)
        sh_mult = eval_sh_bases(self.SH_basis_dim, viewdirs).unsqueeze(
            1).repeat(1, self.config['num_neighbors'], 1)
        agg_sh_feat = torch.sum(sh_mult.unsqueeze(-2) * sh_feat, dim=-1)

        square_dist = (query_points.unsqueeze(
            2) - self.codes_position[indices][:, new_i_torch_gpu, :]).pow(2).sum(dim=-1) + 1e-16

        weight = 1.0 / (torch.sqrt(square_dist) ** self.config['dist_scale'])
        weight = weight / weight.sum(dim=-1, keepdim=True)

        query_codes = torch.matmul(
            weight[0].unsqueeze(1), self.codes[indices][:, new_i_torch_gpu, :][0]).squeeze(1)

        query_sh_codes = torch.matmul(
            weight[0].unsqueeze(1), agg_sh_feat).squeeze(1)

        return query_codes, query_sh_codes

    def get_proposal(self, pts):
        n_rays, n_samples, _ = pts.shape
        new_d_torch_gpu, _ = find_knn(
            self.knn, pts.flatten(0, 1), self.origin_keypoints[0])

        farthest_d = new_d_torch_gpu[:, -1].view(n_rays, n_samples)

        farthest_d[farthest_d > 0.2] = 0.2

        weight = 1.0 / (torch.sqrt(farthest_d) + 1e-10)
        weight = weight / weight.sum(dim=-1, keepdim=True)
        return weight


class SH_IM_Decoder(nn.Module):
    def __init__(self,
                 D=8, W=256, code_dim=64,
                 in_channels_xyz=63, in_channels_dir=27,
                 skips=[4]):
        print(datetime.now().strftime(
            '%Y-%m-%d %H:%M:%S'), 'Building IM-decoder.')
        super(SH_IM_Decoder, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.skips = skips
        self.code_dim = code_dim

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz + self.code_dim, W)
            elif i in skips:
                layer = nn.Linear(W + in_channels_xyz + self.code_dim, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
            nn.Linear(W + in_channels_dir + self.code_dim, W // 2),
            nn.ReLU(True))

        # output layers
        self.sigma = nn.Linear(W, 1)
        self.rgb = nn.Linear(W // 2, 3)

        num_params = sum(p.data.nelement() for p in self.parameters())
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
              'IM decoder done(#parameters=%d).' % num_params)

    def forward(self, codes, sh_codes, xyzdir_embed):
        input_xyz, input_dir = \
            torch.split(xyzdir_embed, [self.in_channels_xyz,
                        self.in_channels_dir], dim=-1)
        input_xyz = torch.cat([codes, input_xyz], dim=-1)
        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        sigma = self.sigma(xyz_)

        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        dir_encoding_input = torch.cat(
            [xyz_encoding_final, sh_codes, input_dir], -1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        rgb = self.rgb(dir_encoding)

        out = torch.cat([rgb, sigma], -1)

        return out


class SHCloudNeRF(nn.Module):
    def __init__(self, keypoints, fps_kps, input_ch, input_ch_views, num_records=1):
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Building network.')
        super().__init__()
        global config
        self.config = config

        self.code_cloud = SHCodeCloud(
            config['code_cloud'], num_records, keypoints, fps_kps)
        self.decoder = SH_IM_Decoder(D=4, W=128, code_dim=config['code_cloud']['code_dim'],
                                  in_channels_xyz=input_ch, in_channels_dir=input_ch_views,
                                  skips=[2])

        num_params = sum(p.data.nelement() for p in self.parameters())
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
              'Network done(#parameters=%d).' % num_params)

    def forward(self, indices, query_points, viewdirs, xyzdir_embedded):
        """
        Args:
            indices: tensor, (batch_size,)
            query_points: tensor, (batch_size, num_points, 3)
        Returns:
            pred_sd: tensor, (batch_size, num_points)
        """
        query_codes, query_sh_codes = self.code_cloud.query(
            indices, query_points[None, ...], viewdirs)
        pred_sd = self.decoder(query_codes, query_sh_codes, xyzdir_embedded)

        return pred_sd


SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
SH_C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
SH_C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]


def eval_sh_bases(basis_dim: int, dirs: torch.Tensor):
    """
    Evaluate spherical harmonics bases at unit directions,
    without taking linear combination.
    At each point, the final result may the be
    obtained through simple multiplication.
    :param basis_dim: int SH basis dim. Currently, 1-25 square numbers supported
    :param dirs: torch.Tensor (..., 3) unit directions
    :return: torch.Tensor (..., basis_dim)
    """
    result = torch.empty(
        (*dirs.shape[:-1], basis_dim), dtype=dirs.dtype, device=dirs.device)
    result[..., 0] = SH_C0
    if basis_dim > 1:
        x, y, z = dirs.unbind(-1)
        result[..., 1] = -SH_C1 * y
        result[..., 2] = SH_C1 * z
        result[..., 3] = -SH_C1 * x
        if basis_dim > 4:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result[..., 4] = SH_C2[0] * xy
            result[..., 5] = SH_C2[1] * yz
            result[..., 6] = SH_C2[2] * (2.0 * zz - xx - yy)
            result[..., 7] = SH_C2[3] * xz
            result[..., 8] = SH_C2[4] * (xx - yy)

            if basis_dim > 9:
                result[..., 9] = SH_C3[0] * y * (3 * xx - yy)
                result[..., 10] = SH_C3[1] * xy * z
                result[..., 11] = SH_C3[2] * y * (4 * zz - xx - yy)
                result[..., 12] = SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
                result[..., 13] = SH_C3[4] * x * (4 * zz - xx - yy)
                result[..., 14] = SH_C3[5] * z * (xx - yy)
                result[..., 15] = SH_C3[6] * x * (xx - 3 * yy)

                if basis_dim > 16:
                    result[..., 16] = SH_C4[0] * xy * (xx - yy)
                    result[..., 17] = SH_C4[1] * yz * (3 * xx - yy)
                    result[..., 18] = SH_C4[2] * xy * (7 * zz - 1)
                    result[..., 19] = SH_C4[3] * yz * (7 * zz - 3)
                    result[..., 20] = SH_C4[4] * (zz * (35 * zz - 30) + 3)
                    result[..., 21] = SH_C4[5] * xz * (7 * zz - 3)
                    result[..., 22] = SH_C4[6] * (xx - yy) * (7 * zz - 1)
                    result[..., 23] = SH_C4[7] * xz * (xx - 3 * yy)
                    result[..., 24] = SH_C4[8] * \
                        (xx * (xx - 3 * yy) - yy * (3 * xx - yy))
    return result

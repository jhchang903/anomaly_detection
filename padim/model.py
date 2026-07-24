import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class FeatureExtractor(nn.Module):
    """Frozen ImageNet resnet18 backbone; returns the layer1/2/3 feature maps
    upsampled to layer1's resolution and concatenated channel-wise (448ch)."""

    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        backbone.eval()
        for param in backbone.parameters():
            param.requires_grad_(False)

        # Stop at layer3 (skip layer4/avgpool/fc) -- PaDiM only needs the
        # intermediate patch-level feature maps, not the classification head.
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

    @torch.no_grad()
    def forward(self, x):
        x = self.stem(x)      # (B, 64, H/4, W/4)
        f1 = self.layer1(x)   # (B, 64, H/4, W/4)
        f2 = self.layer2(f1)  # (B, 128, H/8, W/8)
        f3 = self.layer3(f2)  # (B, 256, H/16, W/16)

        f2 = F.interpolate(f2, size=f1.shape[-2:], mode='nearest')
        f3 = F.interpolate(f3, size=f1.shape[-2:], mode='nearest')

        return torch.cat([f1, f2, f3], dim=1)  # (B, 448, H/4, W/4)


class PaDiM:
    """Statistical anomaly detector -- no backpropagation. `fit()` estimates a
    multivariate Gaussian per patch position from "good" training embeddings;
    `predict()` scores test embeddings by Mahalanobis distance to it."""

    COV_EPS = 0.01  # covariance regularization, as in the PaDiM paper

    def __init__(self, d_reduced=100, feature_dim=448, device='cpu', seed=42):
        self.feature_extractor = FeatureExtractor().to(device)
        self.d_reduced = d_reduced
        generator = torch.Generator().manual_seed(seed)
        self.selected_indices = torch.randperm(feature_dim, generator=generator)[:d_reduced]
        self.mean = None      # (P, d_reduced)
        self.cov_inv = None   # (P, d_reduced, d_reduced)
        self.grid_size = None # (H, W) of the patch grid

    def _embed(self, x, device):
        features = self.feature_extractor(x.to(device))  # (B, 448, H, W)
        b, _, h, w = features.shape
        features = features[:, self.selected_indices.to(device), :, :]
        embeddings = features.permute(0, 2, 3, 1).reshape(b, h * w, self.d_reduced)
        return embeddings, (h, w)

    def fit(self, dataloader, device):
        all_embeddings = []
        for batch in dataloader:
            embeddings, grid_size = self._embed(batch, device)
            self.grid_size = grid_size
            all_embeddings.append(embeddings.cpu())
        embeddings = torch.cat(all_embeddings, dim=0)  # (N, P, d)
        # N = total number of training images
        # P = number of patches per image = h*w; h, w = 64 = 256/4 
        # d = d_reduced = 100

        n = embeddings.shape[0]
        mean = embeddings.mean(dim=0)  # (P, d)
        # 第 0 維(N,也就是「圖片」這個維度)取平均,把 N 這個維度整個「壓掉」

        # 估計每個位置的 共變異矩陣（Covariance）
        centered = embeddings - mean
        cov = torch.einsum('npd,npe->pde', centered, centered) / (n - 1)

        """
        cov = []

        for pixel in range(p):

            X = centered[:, pixel, :]      # (n,d)

            cov_pixel = X.T @ X / (n-1)

            cov.append(cov_pixel)
        """


        cov = cov + self.COV_EPS * torch.eye(self.d_reduced)

        self.mean = mean
        self.cov_inv = torch.linalg.inv(cov)

    def predict(self, x, device, image_size):
        embeddings, (h, w) = self._embed(x, device)
        mean = self.mean.to(device)
        cov_inv = self.cov_inv.to(device)

        delta = embeddings - mean.unsqueeze(0)  # (B, P, d)

        # 計算每個位置的 Mahalanobis Distance，作為異常分數
        dist_sq = torch.einsum('bpd,pde,bpe->bp', delta, cov_inv, delta)

        """
        等同於此寫法
        x = delta[b, p]      # (D,)
        S = cov_inv[p]       # (D,D)

        dist_sq = x.T @ S @ x
        """

        """
        或等同於此寫法
        dist_sq = torch.zeros(B, P)
        for b in range(B):
            for p in range(P):

                x = delta[b, p]          # (D,)
                S = cov_inv[p]           # (D,D)

                dist_sq[b, p] = x @ S @ x
        """


        # to avoid negative values due to numerical errors, clamp to 0 before sqrt
        dist = torch.sqrt(torch.clamp(dist_sq, min=0.0))  # (B, P)

        anomaly_map = dist.reshape(-1, 1, h, w) # (B, 1, H/4, W/4)
        # -1	交給 PyTorch 自動推算這個維度大小,這裡結果會等於 B
        # 1	新增一個 channel 維度,值固定是 1(因為異常分數是純量,不是多 channel 的資料)
        # h, w	把原本攤平的 P 個數字,依序重新排回 h × w 的 2D 網格


        anomaly_map = F.interpolate(anomaly_map, size=image_size, mode='bilinear', align_corners=False)
        # F.interpolate 是為影像設計的函式,強制要求輸入 shape 是 (B, C, H, W)

        anomaly_map = anomaly_map.squeeze(1)  # (B, H, W)

        image_scores = anomaly_map.amax(dim=(1, 2)) # (B,)
        # .amax(dim=(1, 2)) 
        # 表示同時沿著第 1、第 2 兩個維度(也就是 H 和 W)取最大值
        # 把這兩個維度一起壓掉,只留下 B

        return image_scores, anomaly_map

    def save(self, path):
        torch.save({
            'mean': self.mean,
            'cov_inv': self.cov_inv,
            'selected_indices': self.selected_indices,
            'd_reduced': self.d_reduced,
            'grid_size': self.grid_size,
        }, path)

    @classmethod
    def load(cls, path, device='cpu'):
        checkpoint = torch.load(path, map_location=device)
        model = cls(d_reduced=checkpoint['d_reduced'], device=device)
        model.selected_indices = checkpoint['selected_indices']
        model.mean = checkpoint['mean']
        model.cov_inv = checkpoint['cov_inv']
        model.grid_size = checkpoint['grid_size']
        return model

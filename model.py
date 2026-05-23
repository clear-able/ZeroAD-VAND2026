"""
MyAD 模型定义

核心设计：
- 冻结 CLIP ViT-L/14@336px 视觉编码器
- 从 block {12, 15, 18, 21, 24} 提取 5 层特征
- 每层特征经冻结的 ln_post + proj 映射到 768 维
- 每层独立的 AnomalyScorer (768→1，单线性层) 分别处理 CLS 和 patch tokens
- 两路 logit 经 softmax 得到 [正常, 异常] 概率
- 推理时图像级分数 = (1-λ_p)×cls_score + λ_p×max(seg_map)，λ_p=0.5
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clip import load as load_clip


class CLSCrossAttention(nn.Module):
    """
    Cross-attention 用于更新 CLS token。
    CLS 作为 Q，patch tokens 作为 K/V。
    5 层共享同一个实例。
    参数: dim=768（ln_post+proj 后的维度）, num_heads=8, head_dim=64 (inner_dim=512)
    """

    def __init__(self, dim: int = 768, num_heads: int = 8, head_dim: int = 64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, num_heads * head_dim)
        self.k_proj = nn.Linear(dim, num_heads * head_dim)
        self.v_proj = nn.Linear(dim, num_heads * head_dim)
        self.out_proj = nn.Linear(num_heads * head_dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, cls_token, patch_tokens):
        """
        参数:
            cls_token:    [B, 1, D]
            patch_tokens: [B, L, D]
        返回:
            updated_cls: [B, 1, D]（带残差）
        """
        B = cls_token.shape[0]

        cls_normed = self.norm(cls_token)
        patches_normed = self.norm(patch_tokens)

        q = self.q_proj(cls_normed)      # [B, 1, num_heads*head_dim]
        k = self.k_proj(patches_normed)  # [B, L, num_heads*head_dim]
        v = self.v_proj(patches_normed)  # [B, L, num_heads*head_dim]

        q = q.view(B, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)   # [B, H, 1, d]
        k = k.view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, L, d]
        v = v.view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, L, d]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, 1, L]
        attn = F.softmax(attn, dim=-1)
        out = attn @ v  # [B, H, 1, d]

        out = out.permute(0, 2, 1, 3).reshape(B, 1, -1)  # [B, 1, num_heads*head_dim]
        out = self.out_proj(out)  # [B, 1, D]

        return cls_token + out


class AnomalyScorer(nn.Module):
    """
    异常评分头：将特征映射到 1 个 logit。
    结构：Linear(in_planes→1)
    cls 和 seg 分支、各层均独立实例化，不共享参数。
    """

    def __init__(self, in_planes: int = 768):
        super().__init__()
    # (MLP alternative removed for zero-shot)
        self.net = nn.Linear(in_planes, 1)

    def forward(self, x):
        """
        参数:
            x: [B, in_planes] 或 [B*N, in_planes]
        返回:
            logit: [B] 或 [B*N]（squeeze 后）
        """
        return self.net(x).squeeze(-1)


class GlobalSelfAttention(nn.Module):
    """
    全局自注意力：对完整序列（CLS + patch tokens）做自注意力，
    同时更新 CLS 和 patch tokens（参照 AF-CLIP BasicTransformerBlock）。
    5 层共享一个模块。
    """

    def __init__(self, dim: int = 1024, n_heads: int = 8, d_head: int = 64):
        super().__init__()
        inner_dim = n_heads * d_head  # 512
        self.scale = d_head ** -0.5
        self.n_heads = n_heads
        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, seq):
        """
        参数:
            seq: [B, N+1, D]  (CLS + patch tokens)
        返回:
            [B, N+1, D]（带残差，CLS 和 patch 同时更新）
        """
        x = self.norm(seq)
        B, L, _ = x.shape
        h = self.n_heads

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = q.view(B, L, h, -1).permute(0, 2, 1, 3).reshape(B * h, L, -1)
        k = k.view(B, L, h, -1).permute(0, 2, 1, 3).reshape(B * h, L, -1)
        v = v.view(B, L, h, -1).permute(0, 2, 1, 3).reshape(B * h, L, -1)

        attn = torch.einsum('bid,bjd->bij', q, k) * self.scale  # [B*h, L, L]
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bij,bjd->bid', attn, v)             # [B*h, L, d_head]

        out = out.reshape(B, h, L, -1).permute(0, 2, 1, 3).reshape(B, L, -1)  # [B, L, inner_dim]
        out = self.to_out(out)  # [B, L, D]

        return seq + out


class UniADet(nn.Module):
    """
    UniADet 模型

    参数:
        device: 设备 (cuda/cpu)
        features_list: 要提取特征的层索引列表，默认 [12, 15, 18, 21, 24]
        temperature: 温度参数 τ，固定为 0.07
        lambda_p: 推理时 cls 和 seg_max 的融合权重，默认 0.5
    """

    def __init__(
        self,
        device: str = "cuda",
        features_list: list = None,
        lambda_p: float = 0.5,
    ):
        super().__init__()

        self.device = device
        self.features_list = features_list if features_list is not None else [12, 15, 18, 21, 24]
        self.lambda_p = lambda_p
        self.num_layers = len(self.features_list)

        self.clip_model, self.preprocess = load_clip("ViT-L/14@336px", device=device)
        self.clip_model.eval()

        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.feat_dim = self.clip_model.visual.output_dim  # 768

        self.cls_scorers = nn.ModuleList([
            AnomalyScorer(in_planes=self.feat_dim)
            for _ in range(self.num_layers)
        ])
        self.seg_scorers = nn.ModuleList([
            AnomalyScorer(in_planes=self.feat_dim)
            for _ in range(self.num_layers)
        ])

        self.visual_width = self.clip_model.visual.width  # 1024

        self.cls_cross_attn = CLSCrossAttention(dim=self.feat_dim, num_heads=8, head_dim=64)

        self.global_attn = GlobalSelfAttention(dim=self.visual_width)

        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def encode_image(self, image):
        """
        提取图像特征

        参数:
            image: 输入图像 [B, 3, H, W]

        返回:
            cls_features_list: 每层的 CLS token 特征列表，每个 [B, feat_dim=768]
            patch_features_list: 每层的 patch token 特征列表，每个 [B, N, feat_dim=768]
        """
        with torch.no_grad():
            raw_features_list = self.clip_model.encode_image(
                image,
                self.features_list,
                prompt_embeddings=None,
            )

        cls_features_list = []
        patch_features_list = []
        for raw_feat in raw_features_list:
            # raw_feat: [B, N+1, 1024]
            feat_proj = self.clip_model.visual.ln_post(raw_feat.float()) @ self.clip_model.visual.proj
            # feat_proj: [B, N+1, 768]

            cls_feat = F.normalize(feat_proj[:, 0, :], dim=-1)    # [B, 768]
            patch_feat = F.normalize(feat_proj[:, 1:, :], dim=-1)  # [B, N, 768]

            cls_features_list.append(cls_feat)
            patch_features_list.append(patch_feat)

        return cls_features_list, patch_features_list

    def forward_cls(self, cls_features_list):
        """
        计算图像级分类 logits

        参数:
            cls_features_list: 每层的 CLS token 特征列表，每个 [B, feat_dim]

        返回:
            cls_logits_list: 每层的分类 logits 列表，每个 [B, 2]
            cls_probs_list:  每层的分类概率列表，每个 [B, 2]
        """
        cls_logits_list = []
        cls_probs_list = []

        for layer_idx, cls_feat in enumerate(cls_features_list):
            # cls_feat: [B, 768]
            anomaly_logit = self.cls_scorers[layer_idx](cls_feat)  # [B]
            normal_logit = torch.zeros_like(anomaly_logit)          # [B]
            logits = torch.stack([normal_logit, anomaly_logit], dim=-1)  # [B, 2]
            probs = logits

            cls_logits_list.append(logits)
            cls_probs_list.append(probs)

        return cls_logits_list, cls_probs_list

    def forward_seg(self, patch_features_list, image_size):
        """
        计算像素级分割 logits

        参数:
            patch_features_list: 每层的 patch token 特征列表，每个 [B, N, feat_dim]
            image_size: 目标图像尺寸（用于上采样）

        返回:
            seg_logits_list: 每层的分割 logits 列表，每个 [B, 2, H, W]
            seg_probs_list:  每层的分割概率列表，每个 [B, 2, H, W]
        """
        seg_logits_list = []
        seg_probs_list = []

        for layer_idx, patch_feat in enumerate(patch_features_list):
            # patch_feat: [B, N, feat_dim]
            B, N, C = patch_feat.shape
            side = int(N ** 0.5)

            patch_flat = patch_feat.reshape(B * N, C)
            anomaly_logit = self.seg_scorers[layer_idx](patch_flat)   # [B*N]
            normal_logit = torch.zeros_like(anomaly_logit)             # [B*N]
            logits = torch.stack([normal_logit, anomaly_logit], dim=-1)  # [B*N, 2]

            logits = logits.reshape(B, side, side, 2).permute(0, 3, 1, 2)

            logits_upsampled = F.interpolate(
                logits,
                size=(image_size, image_size),
                mode='bilinear',
                align_corners=False,
            )

            probs = logits_upsampled

            seg_logits_list.append(logits_upsampled)
            seg_probs_list.append(probs)

        return seg_logits_list, seg_probs_list

    def forward(self, image, image_size=518):
        """
        前向传播

        参数:
            image: 输入图像 [B, 3, H, W]
            image_size: 目标图像尺寸

        返回:
            cls_logits_list: 每层的分类 logits
            seg_logits_list: 每层的分割 logits（上采样后）
            cls_probs_list: 每层的分类概率
            seg_probs_list: 每层的分割概率
        """
        cls_features_list, patch_features_list = self.encode_image(image)
        cls_logits_list, cls_probs_list = self.forward_cls(cls_features_list)
        seg_logits_list, seg_probs_list = self.forward_seg(patch_features_list, image_size)
        return cls_logits_list, seg_logits_list, cls_probs_list, seg_probs_list

    def inference(self, image, image_size=518):
        """
        推理模式：计算最终的异常分数

        参数:
            image: 输入图像 [B, 3, H, W]
            image_size: 目标图像尺寸

        返回:
            image_scores: 图像级异常分数 [B]
            anomaly_maps: 像素级异常图 [B, H, W]
        """
        self.eval()
        with torch.no_grad():
            cls_logits_list, seg_logits_list, cls_probs_list, seg_probs_list = self.forward(image, image_size)

            cls_scores = torch.stack([probs[:, 1] for probs in cls_probs_list], dim=0).mean(dim=0)  # [B]

            seg_maps = []
            for seg_probs in seg_probs_list:
                # seg_probs: [B, 2, H, W]
                anomaly_map = (seg_probs[:, 1, :, :] + 1 - seg_probs[:, 0, :, :]) / 2.0
                seg_maps.append(anomaly_map)

            anomaly_maps = torch.stack(seg_maps, dim=0).mean(dim=0)  # [B, H, W]

            seg_max_scores = torch.stack([anomaly_map.view(anomaly_map.shape[0], -1).max(dim=-1)[0] for anomaly_map in seg_maps], dim=0).mean(dim=0)  # [B]
            image_scores = (1 - self.lambda_p) * cls_scores + self.lambda_p * seg_max_scores

            # seg_max_scores = anomaly_maps.view(anomaly_maps.shape[0], -1).max(dim=-1)[0]
            # image_scores = (1 - self.lambda_p) * cls_scores + self.lambda_p * seg_max_scores

            # image_scores = cls_scores

            return image_scores, anomaly_maps

    def get_trainable_parameters(self):
        """."""
        params = list(self.cls_scorers.parameters()) + list(self.seg_scorers.parameters())
        return params

    def save_weights(self, path):
        """."""
        state_dict = {
            'cls_scorers': self.cls_scorers.state_dict(),
            'seg_scorers': self.seg_scorers.state_dict(),
            'features_list': self.features_list,
            'feat_dim': self.feat_dim,
            'lambda_p': self.lambda_p,
        }
        torch.save(state_dict, path)

    def load_weights(self, path):
        """."""
        state_dict = torch.load(path, map_location=self.device, weights_only=False)
        self.cls_scorers.load_state_dict(state_dict['cls_scorers'])
        self.seg_scorers.load_state_dict(state_dict['seg_scorers'])


def build_uniadet(device="cuda", features_list=None, **kwargs):
    """
    构建 MyAD 模型

    参数:
        device: 设备
        features_list: 特征提取层列表，默认 [12, 15, 18, 21, 24]
        **kwargs: 其他参数（lambda_p 等）

    返回:
        model: UniADet 模型实例
    """
    if features_list is None:
        features_list = [12, 15, 18, 21, 24]

    model = UniADet(
        device=device,
        features_list=features_list,
        **kwargs
    )


    return model

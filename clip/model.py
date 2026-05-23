"""
CLIP 模型定义（精简版，仅保留 UniADet 所需部分）
从 OpenAI CLIP 官方代码修改而来
"""

from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor, feature_layers=None):
        """
        前向传播，支持提取中间层特征

        参数:
            x: 输入张量 [L, N, D]
            feature_layers: 要提取特征的层索引列表，如 [12, 15, 18, 21, 24]

        返回:
            如果 feature_layers 为 None，返回最终输出
            否则返回指定层的特征列表
        """
        out = []
        for i, block in enumerate(self.resblocks):
            x = block(x)
            if feature_layers is not None and (i + 1) in feature_layers:
                out.append(x)

        if feature_layers is None:
            return x
        else:
            return out


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.width = width
        self.patch_size = patch_size
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, feature_layers=None, prompt_embeddings: torch.Tensor = None):
        """
        前向传播

        参数:
            x: 输入图像 [B, 3, H, W]
            feature_layers: 要提取特征的层索引列表

        返回:
            如果 feature_layers 为 None，返回最终特征 [B, output_dim]
            否则返回中间层特征列表，每个 [B, N+1, output_dim]（已经过 ln_post + proj）
        """
        # Patch embedding
        x = self.conv1(x)  # [B, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, width, grid^2]
        x = x.permute(0, 2, 1)  # [B, grid^2, width]

        # 添加 CLS token
        cls_token = self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls_token, x], dim=1)  # [B, grid^2 + 1, width]

        # 位置编码插值（支持不同输入尺寸）
        side = int((self.positional_embedding.shape[0] - 1) ** 0.5)
        new_side = int((x.shape[1] - 1) ** 0.5)

        if side != new_side:
            pos_embed = self.positional_embedding[1:, :].reshape(1, side, side, -1).permute(0, 3, 1, 2)
            pos_embed = F.interpolate(pos_embed, (new_side, new_side), mode='bilinear', align_corners=False)
            pos_embed = pos_embed.reshape(-1, new_side * new_side).transpose(0, 1)
            pos_embed = torch.cat([self.positional_embedding[:1, :], pos_embed], dim=0)
        else:
            pos_embed = self.positional_embedding

        x = x + pos_embed.to(x.dtype)
        x = self.ln_pre(x)

        prompt_len = 0
        if prompt_embeddings is not None:
            prompt_len = prompt_embeddings.shape[0]
            if prompt_len > 0:
                prompt_tokens = prompt_embeddings.to(dtype=x.dtype, device=x.device)
                prompt_tokens = prompt_tokens.unsqueeze(0).expand(x.shape[0], -1, -1)
                x = torch.cat([x[:, :1, :], prompt_tokens, x[:, 1:, :]], dim=1)

        # Transformer
        x = x.permute(1, 0, 2)  # [N+1, B, width] (NLD -> LND)

        if feature_layers is None:
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # [B, N+1, width]
            if prompt_len > 0:
                x = torch.cat([x[:, :1, :], x[:, 1 + prompt_len:, :]], dim=1)
            x = self.ln_post(x[:, 0, :])  # 只取 CLS token
            x = x @ self.proj
            return x
        else:
            # 提取中间层原始特征（不经过 ln_post + proj）
            # ln_post + proj 由外部在 cross-attention 之后调用
            features = self.transformer(x, feature_layers)
            out = []
            for feat in features:
                feat = feat.permute(1, 0, 2)  # [B, N+1, width=1024]
                if prompt_len > 0:
                    feat = torch.cat([feat[:, :1, :], feat[:, 1 + prompt_len:, :]], dim=1)
                out.append(feat)
            return out


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 image_resolution: int,
                 vision_layers: int,
                 vision_width: int,
                 vision_patch_size: int,
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        vision_heads = vision_width // 64
        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim
        )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, feature_layers=None, prompt_embeddings=None):
        """
        编码图像

        参数:
            image: 输入图像 [B, 3, H, W]
            feature_layers: 要提取特征的层索引列表

        返回:
            如果 feature_layers 为 None，返回图像特征 [B, embed_dim]
            否则返回中间层特征列表
        """
        return self.visual(image.type(self.dtype), feature_layers, prompt_embeddings=prompt_embeddings)

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text


def build_model(state_dict: dict):
    """从 state_dict 构建 CLIP 模型"""
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        raise ValueError("UniADet 只支持 ViT 模型")

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    model.load_state_dict(state_dict)
    return model.eval()

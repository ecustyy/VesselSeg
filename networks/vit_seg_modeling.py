# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from . import vit_seg_configs as configs
from .vit_seg_modeling_resnet_skip import ResNetV2, ParallelDirectionalResNet
from einops import rearrange
import torch.nn.functional as F

logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:   # ResNet
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])  
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ParallelDirectionalResNet(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)

        self.dirtrans = DirectionAwareTransformer(dim=768, num_heads=4)

    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        embeddings = self.dirtrans(x)

        return embeddings, features


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[f"{ROOT}/{ATTENTION_Q}/kernel"]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[f"{ROOT}/{ATTENTION_K}/kernel"]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[f"{ROOT}/{ATTENTION_V}/kernel"]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[f"{ROOT}/{ATTENTION_OUT}/kernel"]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[f"{ROOT}/{ATTENTION_Q}/bias"]).view(-1)
            key_bias = np2th(weights[f"{ROOT}/{ATTENTION_K}/bias"]).view(-1)
            value_bias = np2th(weights[f"{ROOT}/{ATTENTION_V}/bias"]).view(-1)
            out_bias = np2th(weights[f"{ROOT}/{ATTENTION_OUT}/bias"]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[f"{ROOT}/{FC_0}/kernel"]).t()
            mlp_weight_1 = np2th(weights[f"{ROOT}/{FC_1}/kernel"]).t()
            mlp_bias_0 = np2th(weights[f"{ROOT}/{FC_0}/bias"]).t()
            mlp_bias_1 = np2th(weights[f"{ROOT}/{FC_1}/bias"]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[f"{ROOT}/{ATTENTION_NORM}/scale"]))
            self.attention_norm.bias.copy_(np2th(weights[f"{ROOT}/{ATTENTION_NORM}/bias"]))
            self.ffn_norm.weight.copy_(np2th(weights[f"{ROOT}/{MLP_NORM}/scale"]))
            self.ffn_norm.bias.copy_(np2th(weights[f"{ROOT}/{MLP_NORM}/bias"]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded = embedding_output
        attn_weights = None
        return encoded, attn_weights, features


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        final_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        super().__init__(conv2d, upsampling, final_conv)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        # B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        # h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        # x = hidden_states.permute(0, 2, 1)
        # x = x.contiguous().view(B, hidden, h, w)
        x = hidden_states
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        logits = nn.functional.softmax(logits,dim=1)
        return logits

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)


class DirectionalKernelGenerator(nn.Module):
    """可学习的方向核生成器"""

    def __init__(self, kernel_size=3, num_directions=4):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_directions = num_directions
        self.radius = kernel_size // 2

        # 预定义四个主要方向的角度
        self.register_buffer('angles', torch.tensor([0., 45., 90., 135.]))

        # 创建基础网格
        y, x = torch.meshgrid(
            torch.arange(-self.radius, self.radius + 1),
            torch.arange(-self.radius, self.radius + 1)
        )
        self.register_buffer('x', x.float())
        self.register_buffer('y', y.float())

        # 可学习的参数（为每个方向单独学习参数）
        self.sigma = nn.Parameter(torch.ones(num_directions))  # 高斯核的标准差
        self.lambda_param = nn.Parameter(torch.ones(num_directions) * 2.0)  # 波长
        self.psi = nn.Parameter(torch.zeros(num_directions))  # 相位偏移
        self.gamma = nn.Parameter(torch.ones(num_directions) * 0.5)  # 椭圆度

        # 初始化参数
        nn.init.normal_(self.sigma, mean=1.0, std=0.1)
        nn.init.normal_(self.lambda_param, mean=2.0, std=0.3)
        nn.init.normal_(self.psi, mean=0.0, std=0.1)
        nn.init.normal_(self.gamma, mean=0.5, std=0.1)

    def forward(self):
        """生成所有方向核"""
        kernels = []

        for i in range(self.num_directions):
            angle = self.angles[i]
            theta = torch.deg2rad(angle)

            # 旋转坐标
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)

            x_theta = self.x * cos_theta + self.y * sin_theta
            y_theta = -self.x * sin_theta + self.y * cos_theta

            # 椭圆高斯包络
            sigma_x = self.sigma[i]
            sigma_y = sigma_x / torch.clamp(self.gamma[i], min=0.1, max=2.0)

            # Gabor核（实部）
            envelope = torch.exp(-0.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2))
            carrier = torch.cos(2 * np.pi * x_theta / self.lambda_param[i] + self.psi[i])

            gb = envelope * carrier

            # 归一化：使核的和为0，范数为1
            gb = gb - gb.mean()
            gb = gb / (gb.norm() + 1e-6)

            kernels.append(gb.unsqueeze(0))

        # 堆叠所有核：[num_directions, 1, kernel_size, kernel_size]
        kernels = torch.stack(kernels, dim=0)

        return kernels


class DirectionAwareTransformer(nn.Module):
    """方向感知的Transformer模块 - 使用可学习方向核版本"""

    def __init__(self, dim, num_heads=4, kernel_size=3, num_directions=4):
        super().__init__()
        self.dim = dim
        self.num_directions = num_directions
        self.kernel_size = kernel_size

        # 1. 方向核生成器
        self.kernel_generator = DirectionalKernelGenerator(
            kernel_size=kernel_size,
            num_directions=num_directions
        )

        # 2. 可学习的卷积权重（将方向特征映射到原始维度）
        # 输入通道数：num_directions * dim，输出通道数：dim
        self.direction_proj = nn.Conv2d(
            num_directions * dim,
            dim,
            kernel_size=1,
            bias=False
        )

        # 3. 位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, dim, 8, 8) * 0.02)

        # 4. 多头注意力
        self.attn = nn.MultiheadAttention(dim, num_heads)

        # 5. 层归一化
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # 6. FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(0.1)
        )

        # 7. Dropout
        self.dropout = nn.Dropout(0.1)

        # 8. 缩放因子（控制方向特征的强度）
        self.direction_scale = nn.Parameter(torch.tensor(0.1))

    def apply_directional_conv(self, x, kernels):
        """应用方向卷积"""
        B, C, H, W = x.shape
        directional_features = []
        # C = C.to(x.device)
        # 将核扩展到所有通道
        # kernels形状: [num_directions, 1, k, k]
        # 扩展为: [num_directions * C, 1, k, k]
        # 确保kernels与x在同一个设备上
        kernels = kernels.to(x.device)
        kernels_expanded = kernels.repeat_interleave(C, dim=0)

        # 对每个方向应用卷积
        padding = self.kernel_size // 2

        for i in range(self.num_directions):
            # 获取当前方向的核
            start_idx = i * C
            end_idx = (i + 1) * C
            kernel_i = kernels_expanded[start_idx:end_idx]

            # 分组卷积：每个通道使用不同的核
            # 输入: [B, C, H, W]，核: [C, 1, k, k]，groups=C
            dir_feat = F.conv2d(
                x,
                kernel_i,
                padding=padding,
                groups=C
            )
            directional_features.append(dir_feat)

        # 拼接所有方向特征
        # 每个dir_feat: [B, C, H, W]
        # 拼接后: [B, num_directions * C, H, W]
        dir_features = torch.cat(directional_features, dim=1)

        return dir_features

    def forward(self, x):
        B, C, H, W = x.shape

        # ==================== 方向感知分支 ====================
        # 1. 生成方向核
        kernels = self.kernel_generator()  # [num_directions, 1, k, k]

        # 2. 应用方向卷积
        dir_features = self.apply_directional_conv(x, kernels)

        # 3. 投影到原始维度并缩放
        dir_features = self.direction_proj(dir_features)
        dir_features = dir_features * self.direction_scale

        # 4. 与原始特征融合
        x_with_dir = x + self.dropout(dir_features)

        # ==================== Transformer分支 ====================
        # 1. 重排为序列
        x_flat = rearrange(x_with_dir, 'b c h w -> b (h w) c')

        # 2. 位置编码（调整尺寸）
        if H != 8 or W != 8:
            pos_embed_resized = F.interpolate(
                self.pos_embed,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
        else:
            pos_embed_resized = self.pos_embed

        pos_flat = rearrange(pos_embed_resized, 'b c h w -> b (h w) c')
        pos_flat = pos_flat.expand(B, -1, -1)

        # 3. 层归一化
        x_norm = self.norm1(x_flat)

        # 4. 多头注意力（带位置编码）
        # 转置张量以适应 PyTorch 1.8.1 的 MultiheadAttention 输入格式
        x_norm_transposed = (x_norm + pos_flat).transpose(0, 1)  # (seq_len, batch_size, embed_dim)
        value_transposed = x_norm.transpose(0, 1)  # (seq_len, batch_size, embed_dim)

        attn_out, _ = self.attn(
            query=x_norm_transposed,
            key=x_norm_transposed,
            value=value_transposed
        )

        # 转置回原始格式
        attn_out = attn_out.transpose(0, 1)  # (batch_size, seq_len, embed_dim)

        # 5. 残差连接
        x_attn = x_flat + self.dropout(attn_out)

        # 6. FFN
        x_attn_norm = self.norm2(x_attn)
        ffn_out = self.ffn(x_attn_norm)
        x_out = x_attn + self.dropout(ffn_out)

        # 7. 重排回空间格式
        x_out = rearrange(x_out, 'b (h w) c -> b c h w', h=H, w=W)

        return x_out


class MultiScaleDirectionAwareTransformer(nn.Module):
    """多尺度方向感知Transformer"""

    def __init__(self, dim, num_heads=4, num_layers=3):
        super().__init__()

        self.layers = nn.ModuleList([
            DirectionAwareTransformer(
                dim=dim,
                num_heads=num_heads,
                kernel_size=3 + 2 * i  # 不同层使用不同尺度的核
            )
            for i in range(num_layers)
        ])

        # 跨层融合
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * num_layers, dim, kernel_size=1),
            nn.GroupNorm(8, dim),
            nn.GELU()
        )

    def forward(self, x):
        features = []

        for layer in self.layers:
            x = layer(x)
            features.append(x)

        # 融合多尺度特征
        if len(features) > 1:
            fused = torch.cat(features, dim=1)
            x = self.fusion(fused)

        return x


CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}



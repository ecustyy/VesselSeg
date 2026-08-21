import math

from os.path import join as pjoin
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class GaborConv2d(nn.Module):
    """Gabor-like可学习方向卷积"""

    def __init__(self, cin, cout, kernel_size=3, stride=1,
                 padding=1, bias=False, groups=1, num_directions=4, dilation=1):
        super().__init__()
        self.cin = cin
        self.cout = cout
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.num_directions = num_directions
        self.dilation = dilation

        # 标准卷积权重（可学习的Gabor参数）
        self.weight = nn.Parameter(torch.randn(cout, cin // groups, kernel_size, kernel_size))

        # Gabor参数：每个方向有独立的参数
        self.angles = nn.Parameter(torch.tensor([0., 45., 90., 135.]))  # 可学习的方向
        self.sigma = nn.Parameter(torch.ones(num_directions))  # 高斯核标准差
        self.lambda_param = nn.Parameter(torch.ones(num_directions) * 2.0)  # 波长
        self.psi = nn.Parameter(torch.zeros(num_directions))  # 相位

        if bias:
            self.bias = nn.Parameter(torch.zeros(cout))
        else:
            self.register_parameter('bias', None)

        # 初始化网格
        self.radius = kernel_size // 2
        self.register_buffer('grid_x', torch.zeros(kernel_size, kernel_size))
        self.register_buffer('grid_y', torch.zeros(kernel_size, kernel_size))

        # 初始化网格坐标
        with torch.no_grad():
            y, x = torch.meshgrid(
                torch.arange(-self.radius, self.radius + 1),
                torch.arange(-self.radius, self.radius + 1)
            )
            self.grid_x.copy_(x.float())
            self.grid_y.copy_(y.float())

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重为Gabor核"""
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

        # 初始化Gabor参数
        with torch.no_grad():
            self.angles.data = torch.tensor([0., 45., 90., 135.])
            self.sigma.data.fill_(1.0)
            self.lambda_param.data.fill_(2.0)
            self.psi.data.fill_(0.0)

    def generate_gabor_kernels(self):
        """生成Gabor方向核"""
        kernels = []
        angles_rad = torch.deg2rad(self.angles)

        for i in range(self.num_directions):
            theta = angles_rad[i]
            sigma = torch.clamp(self.sigma[i], min=0.5, max=3.0)
            lambda_val = torch.clamp(self.lambda_param[i], min=1.0, max=5.0)
            psi = self.psi[i]

            # 旋转坐标
            x_theta = self.grid_x * torch.cos(theta) + self.grid_y * torch.sin(theta)
            y_theta = -self.grid_x * torch.sin(theta) + self.grid_y * torch.cos(theta)

            # Gabor核（实部）
            gaussian = torch.exp(-0.5 * (x_theta ** 2 + y_theta ** 2) / (sigma ** 2))
            cos_wave = torch.cos(2 * np.pi * x_theta / lambda_val + psi)
            kernel = gaussian * cos_wave

            # 归一化：使核的和为0，L2范数为1
            kernel = kernel - kernel.mean()
            kernel = kernel / (kernel.norm() + 1e-8)

            kernels.append(kernel)

        # 堆叠所有方向核：[num_directions, kernel_size, kernel_size]
        return torch.stack(kernels, dim=0)

    def forward(self, x):
        # 标准化标准卷积权重（保持原有StdConv2d的行为）
        w_std = self.weight
        v, m = torch.var_mean(w_std, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w_std = (w_std - m) / torch.sqrt(v + 1e-5)

        # 生成Gabor方向核
        gabor_kernels = self.generate_gabor_kernels()  # [D, K, K]
        D = gabor_kernels.shape[0]

        # 方案A：将Gabor核集成到标准卷积权重中
        if self.cout >= D:
            # 将Gabor核扩展到与卷积权重相同的形状
            gabor_expanded = gabor_kernels.unsqueeze(1)  # [D, 1, K, K]
            gabor_expanded = gabor_expanded.expand(-1, self.cin // self.groups, -1, -1)

            # 选择前D个输出通道与Gabor核融合
            w_std[:D] = w_std[:D] + 0.1 * gabor_expanded  # 可调节的融合权重

        # 方案B：分别进行标准卷积和方向卷积，然后融合
        # 这里我们先使用方案A，方案B的实现在下面

        # 执行标准卷积
        output = F.conv2d(x, w_std, self.bias, self.stride,
                          self.padding, self.dilation, self.groups)

        return output


def conv3x3_gabor(cin, cout, stride=1, groups=1, bias=False, dilation=1):
    """使用Gabor核的3x3卷积"""
    return GaborConv2d(cin, cout, kernel_size=3, stride=stride,
                       padding=1, bias=bias, groups=groups, dilation=dilation)

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3_gabor(cmid, cmid, stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y

    def load_from(self, weights, n_block, n_unit):
        conv1_weight = np2th(weights[f"{n_block}/{n_unit}/conv1/kernel"], conv=True)
        conv2_weight = np2th(weights[f"{n_block}/{n_unit}/conv2/kernel"], conv=True)
        conv3_weight = np2th(weights[f"{n_block}/{n_unit}/conv3/kernel"], conv=True)

        gn1_weight = np2th(weights[f"{n_block}/{n_unit}/gn1/scale"])
        gn1_bias = np2th(weights[f"{n_block}/{n_unit}/gn1/bias"])

        gn2_weight = np2th(weights[f"{n_block}/{n_unit}/gn2/scale"])
        gn2_bias = np2th(weights[f"{n_block}/{n_unit}/gn2/bias"])

        gn3_weight = np2th(weights[f"{n_block}/{n_unit}/gn3/scale"])
        gn3_bias = np2th(weights[f"{n_block}/{n_unit}/gn3/bias"])

        self.conv1.weight.copy_(conv1_weight)
        self.conv2.weight.copy_(conv2_weight)
        self.conv3.weight.copy_(conv3_weight)

        self.gn1.weight.copy_(gn1_weight.view(-1))
        self.gn1.bias.copy_(gn1_bias.view(-1))

        self.gn2.weight.copy_(gn2_weight.view(-1))
        self.gn2.bias.copy_(gn2_bias.view(-1))

        self.gn3.weight.copy_(gn3_weight.view(-1))
        self.gn3.bias.copy_(gn3_bias.view(-1))

        if hasattr(self, 'downsample'):
            proj_conv_weight = np2th(weights[f"{n_block}/{n_unit}/conv_proj/kernel"], conv=True)
            proj_gn_weight = np2th(weights[f"{n_block}/{n_unit}/gn_proj/scale"])
            proj_gn_bias = np2th(weights[f"{n_block}/{n_unit}/gn_proj/bias"])

            self.downsample.weight.copy_(proj_conv_weight)
            self.gn_proj.weight.copy_(proj_gn_weight.view(-1))
            self.gn_proj.bias.copy_(proj_gn_bias.view(-1))

class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))


        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width*4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width)) for i in range(2, block_units[0] + 1)],
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2)) for i in range(2, block_units[1] + 1)],
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4)) for i in range(2, block_units[2] + 1)],
                ))),
        ]))

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body)-1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i+1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        return x, features[::-1]


class ParallelDirectionalResNet(nn.Module):
    """并行多方向卷积的ResNetV2"""

    def __init__(self, block_units, width_factor, num_directions=4):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width
        self.num_directions = num_directions

        # 原始卷积路径
        self.original_conv = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width // 2, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width // 2, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
        ]))

        # 多方向卷积路径
        self.directional_convs = nn.ModuleList()
        directions = [0, 45, 90, 135][:num_directions]

        for dir_angle in directions:
            conv = nn.Sequential(OrderedDict([
                ('conv', StdConv2d(3, width // (2 * num_directions),
                                   kernel_size=7, stride=2, bias=False, padding=3)),
                ('gn', nn.GroupNorm(8, width // (2 * num_directions), eps=1e-6)),
                ('relu', nn.ReLU(inplace=True)),
            ]))
            # 初始化方向卷积核
            self._init_directional_kernel(conv[0].weight, dir_angle)
            self.directional_convs.append(conv)

        # 合并层
        self.merge = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(width, width, kernel_size=1, bias=False)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
        ]))

        # 保持原有的body部分
        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width * 4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width * 4, cout=width * 4, cmid=width)) for i in
                 range(2, block_units[0] + 1)],
            ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width * 4, cout=width * 8, cmid=width * 2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width * 8, cout=width * 8, cmid=width * 2)) for i in
                 range(2, block_units[1] + 1)],
            ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width * 8, cout=width * 16, cmid=width * 4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width * 16, cout=width * 16, cmid=width * 4)) for i in
                 range(2, block_units[2] + 1)],
            ))),
        ]))

    def _init_directional_kernel(self, weight, angle_deg):
        """初始化方向卷积核"""
        with torch.no_grad():
            kernel_size = weight.shape[-1]
            center = kernel_size // 2

            for out_c in range(weight.shape[0]):
                for in_c in range(weight.shape[1]):
                    kernel = torch.zeros(kernel_size, kernel_size)
                    angle_rad = math.radians(angle_deg)

                    # 创建方向滤波器
                    for i in range(kernel_size):
                        for j in range(kernel_size):
                            dx = j - center
                            dy = i - center

                            # 计算该点与目标方向的一致性
                            if dx == 0 and dy == 0:
                                val = 1.0
                            else:
                                point_angle = math.atan2(dy, dx)
                                angle_diff = abs(point_angle - angle_rad)
                                cos_sim = math.cos(angle_diff)
                                val = max(0, cos_sim ** 2)  # 平方增强方向性

                            kernel[i, j] = val

                    # 归一化并添加随机噪声
                    kernel = kernel / (kernel.sum() + 1e-8)
                    kernel = kernel + torch.randn_like(kernel) * 0.01
                    weight[out_c, in_c] = kernel

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()

        # 原始卷积路径
        orig_out = self.original_conv(x)

        # 多方向卷积路径
        dir_outputs = []
        for conv in self.directional_convs:
            dir_out = conv(x)
            dir_outputs.append(dir_out)

        # 合并所有特征
        if dir_outputs:
            all_dir = torch.cat(dir_outputs, dim=1)
            combined = torch.cat([orig_out, all_dir], dim=1)
        else:
            combined = orig_out

        # 融合特征
        x = self.merge(combined)
        features.append(x)

        # 继续原有流程
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)

        for i in range(len(self.body) - 1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i + 1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)

        x = self.body[-1](x)
        return x, features[::-1]

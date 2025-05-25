import torch.nn as nn
from ultralytics.nn.modules import C3
from ultralytics.nn.modules.block import RepBottleneck
from timm.models.layers import SqueezeExcite
import torch
# https://github.com/THU-MIG/RepViT/tree/main
# https://arxiv.org/pdf/2307.09283
# https://arxiv.org/abs/2312.05760

__all__=['RepViTBlock','C2f_RepViTBlock','C3k2_RepViTBlock']
'''
                            CVPR 2024 顶会
即插即用模块：RepViTBlock
最近，与轻量级卷积神经网络 （CNN） 相比，轻量级 Vision Transformer （ViTs） 
在资源受限的移动设备上表现出卓越的性能和更低的延迟。研究人员发现了轻量级 ViT 和轻量级 CNN 之间的许多结构联系。
然而，它们之间在块结构、宏观和微观设计方面的显着架构差异尚未得到充分检验。在这项研究中，
我们从 ViT 的角度重新审视了轻量级 CNN 的高效设计，并强调了它们在移动设备中的前景。
具体来说，我们通过集成轻量级 ViT 的高效架构设计，逐步增强了标准轻量级 CNN（即 MobileNetV3）的移动友好性。
这最终产生了一个新的纯轻量级 CNN 系列，即 RepViT。通过广泛的实验表明，RepViT 的性能优于现有的最先进的轻量级 ViT，
并且在各种视觉任务中表现出良好的性能。

RepViTBlock 模块的作用总结如下：
1.分离 Token Mixer 和 Channel Mixer：
RepViTBlock 模块将 token mixer（通过 3x3 深度卷积处理空间信息）和 channel mixer（通过 1x1 卷积处理通道信息）分离开来。
这种设计提高了空间信息与通道信息的处理效率，受到了 Vision Transformer (ViT) 架构的启发。
2.结构重参数化：
RepViTBlock 在训练阶段使用结构重参数化技术，通过多分支结构增强模型的学习能力，
而在推理阶段简化网络结构，减少计算和内存开销。这种技术特别适合在移动设备上应用，有效降低了延迟和资源消耗。
3.提升移动设备效率：
RepViTBlock 能在保持高准确率的同时，大幅度提升模型在移动设备上的推理效率，

适用于图像分类、目标检测、实例分割、语义分割等所有计算机视觉CV任务
'''

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class Bottleneck_RepViTBlock(Bottleneck):
    """Standard bottleneck with DCNV3."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv2 = RepViTBlock(c_, c2, k[1])

class C2f_RepViTBlock(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_RepViTBlock(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))
class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

class C3k2_RepViTBlock(C2f_RepViTBlock):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_RepViTBlock(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n)
        )

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups,
                            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)

    @torch.no_grad()
    def fuse(self):
        if isinstance(self.m, Conv2d_BN):
            m = self.m.fuse()
            assert (m.groups == m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        elif isinstance(self.m, torch.nn.Conv2d):
            m = self.m
            assert (m.groups != m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self


class RepVGGDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = torch.nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed
        self.bn = torch.nn.BatchNorm2d(ed)

    def forward(self, x):
        return self.bn((self.conv(x) + self.conv1(x)) + x)

    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [1, 1, 1, 1])

        identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device),
                                           [1, 1, 1, 1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = conv.weight * w[:, None, None, None]
        b = bn.bias + (conv.bias - bn.running_mean) * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)
        return conv
class RepViTBlock(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, stride=1, use_se=True, use_hs=True):
        super(RepViTBlock, self).__init__()
        hidden_dim = 2*inp
        if stride == 2:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                Conv2d_BN(inp, oup, ks=1, stride=1, pad=0)
            )
            self.channel_mixer = Residual(nn.Sequential(
                # pw
                Conv2d_BN(oup, 2 * oup, 1, 1, 0),
                nn.GELU() if use_hs else nn.GELU(),
                # pw-linear
                Conv2d_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0),
            ))
        else:
            self.token_mixer = nn.Sequential(
                RepVGGDW(inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                Conv2d_BN(inp, oup, ks=1, stride=1, pad=0)
            )
            self.channel_mixer = Residual(nn.Sequential(
                # pw
                Conv2d_BN(oup, hidden_dim, 1, 1, 0),
                nn.GELU() if use_hs else nn.GELU(),
                # pw-linear
                Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
            ))

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))

if __name__ == '__main__':
    # 定义输入张量的形状为 B, C, H, W
    input = torch.randn(1, 32, 64, 64)
    # 创建一个 RepViTBlock 模块实例
    model = RepViTBlock(inp=32,oup=32,kernel_size=3, stride=2)
    # 执行前向传播
    output = model(input)
    # 打印输入和输出的形状
    print('input_size:',input.size())
    print('output_size:',output.size())

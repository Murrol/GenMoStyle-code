import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from collections import OrderedDict
import math
# import numpy as np

# def init_linear(linear):
#     nn.init.xavier_normal(linear.weight)
#     if linear.bias is not None:
#         linear.bias.data.zero_()
#
# def init_conv(conv):
#     nn.init.kaiming_normal(conv.weight)
#     if conv.bias is not None:
#         conv.bias.data.zero_()

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, margin=3.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

def init_weight(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_normal_(m.weight)
        # m.bias.data.fill_(0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def gan_loss(pred, should_be_classified_as_real):
    bs = pred.size(0)
    if should_be_classified_as_real:
        return F.softplus(-pred).mean()
    else:
        return F.softplus(pred).mean()

def normalization(v):
    if type(v) == list:
        return [normalization(vv) for vv in v]
    return v * torch.rsqrt((torch.sum(v ** 2, dim=1, keepdim=True) + 1e-8))

class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias)
        # linear.weight.data.normal_()
        # linear.bias.data.zero_()
        # self.linear = equal_lr(linear)
        init_weight(self.linear)

    def forward(self, input):
        return self.linear(input)


class EqualConv1d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv1d(*args, **kwargs)
        # conv.weight.data.normal_()
        # if conv.bias is not None:
        #     conv.bias.data.zero_()
        # self.conv = equal_lr(conv)
        init_weight(self.conv)

    def forward(self, input):
        return self.conv(input)


class AdaptiveInstanceNorm1d(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.InstanceNorm1d(in_channel)
        self.style_net = EqualLinear(style_dim, in_channel*2)

        self.style_net.linear.bias.data[:in_channel] = 1
        self.style_net.linear.bias.data[in_channel:] = 0

    def forward(self, input, style):
        # input (B, C, seq_len)
        # style (B, style_dim) -> (B, C * 2, 1)
        # print("StyleBlock: Style INF", style.isinf().sum().item())
        # in_s = style

        style = self.style_net(style).unsqueeze(2)
        # if style.isnan().sum().item() > 0:
            # print(in_s)
            # print(self.style_net.linear.weight)
            # print(self.style_net.linear.bias)
            # print(style)
        # print(math.prod(style.shape))
        gamma, beta = style.chunk(2, 1)
        out = self.norm(input)
        # print("StyleBlock: Style", style.isnan().sum().item())
        # print("StyleBlock: Input", input.isnan().sum().item())
        # print("StyleBlock: Gamma", gamma.isnan().sum().item())
        # print("StyleBlock: Beta", beta.isnan().sum().item())
        # print("StyleBlock: out", out.isnan().sum().item())
        out = gamma * out + beta
        # print("StyleBlock: out INF", out.isinf().sum().item())
        return out


class Conv1dLayer(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size, downsample=False, bias=True, drop_prob=0, activate=True, norm="none"):
        super().__init__()
        layers = []

        padding = kernel_size // 2
        if padding > 0:
            layers.append(("RefPad", nn.ReflectionPad1d(padding)))
            padding = 0

        if downsample:
            # factor = 2
            stride = 2
        else:
            stride = 1

        layers.append(("Conv",
                      EqualConv1d(
                          in_channel,
                          out_channel,
                          kernel_size,
                          padding=padding,
                          stride=stride,
                          bias = bias
                      )))
        if norm == "bn":
            layers.append(("BatchNorm", nn.BatchNorm1d(out_channel)))
        elif norm == "in":
            layers.append(("InstanceNorm", nn.InstanceNorm1d(out_channel)))
        elif norm == "none":
            pass
        else:
            assert 0, "Unsupported normalization:{}".format(norm)

        if drop_prob>0:
            layers.append(("Drop", nn.Dropout(drop_prob, inplace=True)))

        if activate:
            layers.append(("Act", nn.LeakyReLU(0.2, inplace=True)))
        # self.model = nn.Sequential(layers)
        super().__init__(OrderedDict(layers))

    def forward(self, input):
        out = super().forward(input)
        return out

class StyleConv1dLayer(nn.Module):
    def __init__(self, in_channel, out_channel, style_dim, upsample=True):
        super().__init__()

        if upsample:
            self.conv1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                EqualConv1d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
            )
        else:
            self.conv1 = EqualConv1d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)

        self.adain1 = AdaptiveInstanceNorm1d(out_channel, style_dim)
        self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, input, style):
        out = self.conv1(input)
        out = self.lrelu1(out)
        out = self.adain1(out, style)
        return out

class SimpleConv1dLayer(nn.Module):
    def __init__(self, in_channel, out_channel, upsample=True):
        super().__init__()

        if upsample:
            self.conv1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                EqualConv1d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
            )
        else:
            self.conv1 = EqualConv1d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)

        self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, input):
        out = self.conv1(input)
        out = self.lrelu1(out)
        return out

class ShallowResBlock(nn.Module):
    def __init__(self, channel, norm="none"):
        super().__init__()
        self.conv = Conv1dLayer(in_channel=channel, out_channel=channel, kernel_size=3,
                                norm=norm, activate=False, bias=False)

    def forward(self, x):
        out = self.conv(x)
        out += x
        return out

class DownsampleResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = Conv1dLayer(in_channel, in_channel, kernel_size=3, downsample=False)
        self.conv2 = Conv1dLayer(in_channel, out_channel, kernel_size=3, downsample=True)

        self.conv3 = Conv1dLayer(in_channel, out_channel, kernel_size=3, downsample=True,
                                 activate=False, bias=False)

    def forward(self, input):
        # print(input.shape)
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.conv3(input)
        out = (out+skip) / math.sqrt(2)
        # print(out.shape)
        return out

# class Upsampler(nn.Module):
#     def __init__(self, in_channel, out_channel, upsample=True):
#         super().__init__()
#
#         if upsample:
#             self.conv1 = nn.Sequential(
#                 nn.Upsample(scale_factor=2, mode="nearest"),
#                 EqualConv1d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
#             )
#         else:
#             self.conv1 = EqualConv1d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
#
#         self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)
#
#     def forward(self, input):
#         out = self.conv1(input)
#         out = self.lrelu1(out)
#         return out
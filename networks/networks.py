import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
import numpy as np
from networks.blocks import *


def reparametrize(mu, logvar):
    s_var = logvar.mul(0.5).exp_()
    eps = s_var.data.new(s_var.size()).normal_()
    return eps.mul(s_var).add_(mu)

class StyleContentEncoder(nn.Module):
    def __init__(self, mid_channels, sp_channels, st_channels):
        super().__init__()
        self.content_encoder = ContentEncoder(mid_channels, sp_channels)
        self.style_encoder = StyleEncoder(mid_channels, st_channels)

    def forward(self, input, action_vecs=None, style_vecs=None):
        sp = self.content_encoder(input, action_vecs)
        gl_mu, gl_logvar = self.style_encoder(input, style_vecs)
        return sp, gl_mu, gl_logvar

    def extract_content_feature(self, input, action_vecs=None):
        return self.content_encoder(input, action_vecs)

    def extract_style_feature(self, input, style_vecs=None):
        return self.style_encoder(input, style_vecs)

class ContentEncoder(nn.Module):
    def __init__(self, mid_channels, sp_channels):
        super().__init__()
        # channels = [263, 512, 1024, 1024]
        # scale = [96, 48, 24, 12]
        n_down = len(mid_channels) - 1
        self.ToMidPoint = []
        for i in range(1, n_down+1):
            self.ToMidPoint.append(
                Conv1dLayer(mid_channels[i - 1], mid_channels[i], kernel_size=3, downsample=True, norm="in"))
        self.ToMidPoint = nn.Sequential(*self.ToMidPoint)

        # sp_channels = [1024, 512, 256]
        ToSpatialCode = []
        for i in range(len(sp_channels)-1):
            ToSpatialCode.append(
                Conv1dLayer(sp_channels[i], sp_channels[i+1], kernel_size=1, norm="in"))
        ToSpatialCode.append(Conv1dLayer(sp_channels[-1], sp_channels[-1], kernel_size=1, activate=False))
        self.ToSpatialCode = nn.Sequential(*ToSpatialCode)

    def forward(self, input, action_vecs=None):
        midpoint = self.ToMidPoint(input)

        B, S, L = midpoint.shape
        if action_vecs is not None:
            action_vecs = action_vecs.unsqueeze(-1).repeat(1, 1, L)
            sp_input = torch.cat([midpoint, action_vecs], dim=1)
        else:
            sp_input = midpoint

        sp = self.ToSpatialCode(sp_input)
        return sp

class StyleEncoder(nn.Module):
    def __init__(self, mid_channels, st_channels):
        super().__init__()
        # channels = [263, 512, 1024, 1024]
        # scale = [96, 48, 24, 12]
        n_down = len(mid_channels) - 1
        self.ToMidPoint = []
        for i in range(1, n_down+1):
            self.ToMidPoint.append(
                Conv1dLayer(mid_channels[i - 1], mid_channels[i], kernel_size=3, downsample=True)
            )
        self.ToMidPoint = nn.Sequential(*self.ToMidPoint)

        # st_channels = [1024, ]
        ToGlobalCode = []
        for i in range(len(st_channels)-1):
            ToGlobalCode.append(
                Conv1dLayer(st_channels[i], st_channels[i+1], kernel_size=3, downsample=True)
            )
        ToGlobalCode.append(nn.AdaptiveAvgPool1d(1))
        ToGlobalCode.append(Conv1dLayer(st_channels[-1], st_channels[-1]*2, kernel_size=1, activate=False))
        self.ToGlobalCode = nn.Sequential(*ToGlobalCode)

    def forward(self, input, style_vecs=None):
        midpoint = self.ToMidPoint(input)
        B, S, L = midpoint.shape

        if style_vecs is not None:
            style_vecs = style_vecs.unsqueeze(-1).repeat(1, 1, L)
            gl_input = torch.cat([midpoint, style_vecs], dim=1)
        else:
            gl_input = midpoint

        gl = self.ToGlobalCode(gl_input).squeeze(-1)
        gl_mu, gl_logvar = gl.chunk(2, 1)
        return gl_mu, gl_logvar


class Generator(nn.Module):
    def __init__(self, n_conv, n_up, dim_pose, channels, style_dim):
        super().__init__()
        self.layers = nn.ModuleList()
        # self.n_up = len(channels) - 1
        # 32 -> 64 -> 128 -> 256
        # 512 -> 1024 -> 512 -> 263
        for i in range(n_conv):
            self.layers.append(StyleConv1dLayer(channels[i], channels[i+1], style_dim, upsample=False))
        for i in range(n_up):
            self.layers.append(StyleConv1dLayer(channels[i+n_conv], channels[i+n_conv+1], style_dim))

        self.out_linear = Conv1dLayer(channels[n_conv+n_up], dim_pose, kernel_size=1, activate=False)

    def forward(self, input, style, action_vecs=None, style_vecs=None):
        # input =
        # out = inWGput
        B, SI, L = input.shape
        # print(input.isnan().sum().item())
        # print(style.isnan().sum().item())
        # B, SS = style.shape
        if action_vecs is not None:
            action_vecs = action_vecs.unsqueeze(-1).repeat(1, 1, L)
            sp_input = torch.cat([input, action_vecs], dim=1)
        else:
            sp_input = input

        if style_vecs is not None:
            # style_vecs = style_vecs.unsqueeze(-1).repeat(1, 1, L)
            gl_input = torch.cat([style, style_vecs], dim=1)
        else:
            gl_input = style

        # print(sp_input.isinf().sum().item())
        # print(gl_input.isinf().sum().item())
        # print('------------------------')
        for i in range(len(self.layers)):
            # print(sp_input.shape)
            # print(gl_input.shape)
            # print(input.shape, style.shape)
            # print(i, sp_input.isinf().sum().item())
            # # if sp_input
            # print(i, gl_input.isinf().sum().item())
            sp_input = self.layers[i](sp_input, gl_input)

        # print('------------------------')
        # print(input.shape)
        output = self.out_linear(sp_input)
        return output


class GlobalRegressor(nn.Module):
    def __init__(self, n_conv, dim_out, channels):
        super().__init__()
        layers = []
        for i in range(n_conv):
            layers.append(Conv1dLayer(channels[i], channels[i+1], kernel_size=3, downsample=False))
        # layers.append(nn.Dropout(p_drop))
        layers.append(Conv1dLayer(channels[n_conv], dim_out, kernel_size=1, activate=False, bias=False, downsample=False))
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)


class MotionEncoder(nn.Module):
    def __init__(self, channels, output_size, vae_encoder=False):
        super().__init__()
        # channels = [263, 512, 1024, 1024]
        # scale = [96, 48, 24, 12]
        n_down = len(channels) - 1
        # self.ToMidPoint = []
        model = []
        for i in range(1, n_down+1):
            model.append(
                Conv1dLayer(channels[i-1], channels[i], kernel_size=3, drop_prob=0.2, downsample=True)
            )
        if vae_encoder:
            model.append(Conv1dLayer(channels[-1], output_size*2, kernel_size=1, activate=False))
        else:
            model.append(Conv1dLayer(channels[-1], output_size, kernel_size=1, activate=False))
        self.model = nn.Sequential(*model)
        self.vae_encoder = vae_encoder

    def forward(self, input):
        output = self.model(input)
        if self.vae_encoder:
            mean, logvar = output.chunk(2, 1)
            return reparametrize(mean, logvar), mean, logvar
        else:
            return output, None, None
        # return sp_mu, sp_logvar


class MotionDecoder(nn.Module):
    def __init__(self, channels, output_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # self.n_up = len(channels) - 1
        # 32 -> 64 -> 128 -> 256
        # 512 -> 1024 -> 512 -> 263
        n_up = len(channels) - 1
        model = []
        model.append(Conv1dLayer(channels[0], channels[0], kernel_size=3, downsample=False))
        for i in range(n_up):
            model.append(SimpleConv1dLayer(channels[i], channels[i+1], upsample=True))

        model.append(Conv1dLayer(channels[-1], output_size, kernel_size=1, activate=False, downsample=False))
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class ResNetClassifier(nn.Module):
    def __init__(self, mid_channels, st_channels, num_classes):
        super().__init__()
        # channels = [263, 512, 1024, 1024]
        # scale = [96, 48, 24, 12]
        n_down = len(mid_channels) - 1
        self.ToMidPoint = []
        for i in range(1, n_down+1):
            self.ToMidPoint.append(
                Conv1dLayer(mid_channels[i-1], mid_channels[i], kernel_size=3, downsample=True)
            )
        self.ToMidPoint = nn.Sequential(*self.ToMidPoint)

        # st_channels = [1024, ]
        self.ToGlobalCode = nn.Sequential(
            Conv1dLayer(st_channels[0], st_channels[1], kernel_size=3, downsample=True),
            Conv1dLayer(st_channels[1], st_channels[2], kernel_size=3, downsample=True),
            nn.AdaptiveAvgPool1d(1),
            Conv1dLayer(st_channels[2], st_channels[2], kernel_size=1, activate=False, bias=True)
        )
        self.output = EqualLinear(st_channels[2], num_classes, bias=False)

    def forward(self, input):
        # print(input.shape)
        # print("Encoder", input.isnan().sum().item())

        # mask = torch.ones_like(input, device=input.device)
        # mask[:, :3] *= 0
        # input = input * mask
        midpoint = self.ToMidPoint(input)
        # print(midpoint.shape)
        gl = self.ToGlobalCode(midpoint).squeeze(-1)
        # print(sp.shape, gl.shape)
        # sp = normalization(sp)
        # gl = F.sigmoid(gl)
        gl = normalization(gl)
        # gl_mu, gl_logvar = gl.chunk(2, 1)
        pred = self.output(gl)
        return gl, pred


class GRUClassifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.input_emb = EqualLinear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.hidden = nn.Parameter(torch.randn(2, 1,hidden_size), requires_grad=True)
        self.out_emb = nn.Sequential(
            EqualLinear(hidden_size*2, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.output = EqualLinear(256, output_size)

    def forward(self, inputs):
        embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, len(embs), 1)
        _, gru_last = self.gru(embs, hidden)
        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)
        pred_feat = self.out_emb(gru_last)
        pred_feat = normalization(pred_feat)
        return pred_feat, self.output(pred_feat)


class ResNetDisAndCls(nn.Module):
    def __init__(self, mid_channels, sp_channels, st_channels, num_classes, num_digits):
        super().__init__()
        # channels = [263, 512, 1024, 1024]
        # scale = [96, 48, 24, 12]
        n_down = len(mid_channels) - 1
        self.ToMidPoint = []
        for i in range(1, n_down+1):
            self.ToMidPoint.append(
                Conv1dLayer(mid_channels[i - 1], mid_channels[i], kernel_size=3, downsample=True)
            )
        self.ToMidPoint = nn.Sequential(*self.ToMidPoint)

        # sp_channels = [1024, 512, 256]
        self.ToSpatialCode = nn.Sequential(
            Conv1dLayer(sp_channels[0], sp_channels[1], kernel_size=1),
            Conv1dLayer(sp_channels[1], sp_channels[2], kernel_size=1)
        )
        self.OutDis = Conv1dLayer(sp_channels[2], num_digits, kernel_size=1, activate=False, bias=False)

        # st_channels = [1024, ]
        self.ToGlobalCode = nn.Sequential(
            Conv1dLayer(st_channels[0], st_channels[1], kernel_size=3, downsample=True),
            Conv1dLayer(st_channels[1], st_channels[2], kernel_size=3, downsample=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.OutCls = EqualLinear(st_channels[2], num_classes, bias=False)

    def forward(self, input):
        # print(input.shape)
        # print("Encoder", input.isnan().sum().item())

        midpoint = self.ToMidPoint(input)
        # print(midpoint.shape)
        sp = self.ToSpatialCode(midpoint)

        gl = self.ToGlobalCode(midpoint).squeeze()

        dis_pred = self.OutDis(sp)
        cls_pred = self.OutCls(gl)
        return sp, dis_pred, gl, cls_pred

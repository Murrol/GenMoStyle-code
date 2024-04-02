import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gcn import GraphConv


###############################################################################
# Helper Functions
###############################################################################
def init_weights(net):
    def init_func(module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    print('Initializing %s...' % net._get_name())
    net.apply(init_func)


def init_net(net):
    # TODO: Multi-GPUs
    # if len(gpu_ids) > 0: 
    #     assert(torch.cuda.is_available())
    #     net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net)
    return net


def define_AE(input_nc, output_nc, ngf):
    netAE = Generator(input_nc=input_nc, output_nc=output_nc, ngf=ngf)
    return init_net(netAE)


def define_G(input_nc, output_nc, ngf, style_dim, n_blk, n_btn):
    netG = Generator(input_nc=input_nc, output_nc=output_nc, ngf=ngf, style_dim=style_dim, n_blk=n_blk, n_btn=n_btn)
    return init_net(netG)


def define_F(latent_dim, hidden_dim, style_dim, num_domains):
    netF = MappingNetwork(latent_dim=latent_dim, hidden_dim=hidden_dim, style_dim=style_dim, num_domains=num_domains)
    return init_net(netF)


def define_E(input_nc, nef, style_dim, num_domains, clip_size, n_blk, n_btn):
    netE = StyleEncoder(input_nc=input_nc, nef=nef, style_dim=style_dim, num_domains=num_domains, clip_size=clip_size, n_blk=n_blk, n_btn=n_btn)
    return init_net(netE)


def define_D(input_nc, ndf, num_domains, clip_size, n_blk, n_btn):
    netD = Discriminator(input_nc=input_nc, ndf=ndf, num_domains=num_domains, clip_size=clip_size, n_blk=n_blk, n_btn=n_btn)
    return init_net(netD)


def print_network(net, out_f=None):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    if out_f is not None:
        out_f.write(net.__repr__() + "\n")
        out_f.write('Total number of parameters: %d\n' % num_params)
        out_f.flush()


###############################################################################
# ST-GCN based Basic Modules
###############################################################################
class ST_Conv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride=1, downsample=False, bias=True):
        super(ST_Conv, self).__init__()
        assert len(kernel_size) == 2  # TODO: (3,3)?
        self.downsample = downsample
        t_stride = stride
        t_padding = (kernel_size[0] - 1) // 2
        self.s_conv = GraphConv(dim_in,
                                dim_out,
                                s_kernel_size=kernel_size[1],
                                bias=bias)
        self.t_conv = nn.Conv2d(dim_out,
                                dim_out,
                                kernel_size=(kernel_size[0], 1),
                                stride=(t_stride, 1),
                                padding=(t_padding, 0),
                                bias=bias)
        if downsample:
            self.s_avgpool = S_AvgPool()
            self.t_avgpool = T_AvgPool()
        self.actv = nn.ReLU(inplace=True)

    def forward(self, x, A, M=None):
        x, A = self.s_conv(x, A)
        if self.downsample:
            x = self.s_avgpool(x, M)
            x = self.t_avgpool(x)
        x = self.actv(x)
        x = self.t_conv(x)
        return x


class ST_ConvTranspose(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride=1, upsample=False, bias=True):
        super(ST_ConvTranspose, self).__init__()
        assert len(kernel_size) == 2
        self.upsample = upsample
        t_stride = stride
        t_padding = (kernel_size[0] - 1) // 2
        self.s_conv = GraphConv(dim_in,
                                dim_out,
                                s_kernel_size=kernel_size[1],
                                bias=bias)
        self.t_conv = nn.Conv2d(dim_out,
                                dim_out,
                                kernel_size=(kernel_size[0], 1),
                                stride=(t_stride, 1),
                                padding=(t_padding, 0))
        if upsample:
            self.s_unpool = S_AvgUnpool()
            self.t_unpool = T_AvgUnpool()
        self.actv = nn.ReLU(inplace=True)

    def forward(self, x, A, M=None):
        x, A = self.s_conv(x, A)
        if self.upsample:
            x = self.s_unpool(x, M)
            x = self.t_unpool(x)
        x = self.actv(x)
        x = self.t_conv(x)
        return x


###############################################################################
# Down/Up-sampling
###############################################################################
class S_AvgPool(nn.Module):
    def __init__(self):
        super(S_AvgPool, self).__init__()

    def forward(self, x, M):
        x = torch.einsum('nctv,vw->nctw', x, M)
        return x.contiguous()


class S_AvgUnpool(nn.Module):
    def __init__(self):
        super(S_AvgUnpool, self).__init__()

    def forward(self, x, M):
        x = torch.einsum('nctv,vw->nctw', x, M)
        return x.contiguous()


class T_AvgPool(nn.Module):
    def __init__(self):
        super(T_AvgPool, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=(2, 1))

    def forward(self, x):
        x = self.avgpool(x)
        return x


class T_AvgUnpool(nn.Module):
    def __init__(self):
        super(T_AvgUnpool, self).__init__()
        self.unpool = F.interpolate

    def forward(self, x):
        x = self.unpool(x, scale_factor=(2, 1), mode='bilinear', align_corners=True)
        return x


###############################################################################
# ST-GCN based Resnet Blocks
###############################################################################
class ST_ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False, highpass=True):
        super(ST_ResBlk, self).__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.highpass = highpass
        self.projection = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = ST_Conv(dim_in, dim_in, kernel_size=(3, 3), downsample=self.downsample)  # where temporal dimension reduction occurs
        self.conv2 = ST_Conv(dim_in, dim_out, kernel_size=(3, 3))
        if self.projection:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.downsample:
            self.s_avgpool = S_AvgPool()
            self.t_avgpool = T_AvgPool()

    def _shortcut(self, x, M):
        if self.projection:
            x = self.conv1x1(x)
        if self.downsample:
            x = self.s_avgpool(x, M)
            x = self.t_avgpool(x)
        return x

    def _residual(self, x, A30, A31, M):
        A3 = A30 if A30 is not None else A31
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x, A3, M)
        if self.downsample:
            A3 = A31

        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x, A3)
        return x

    def forward(self, x, A30=None, A31=None, M=None):
        if self.highpass:
            x = self._shortcut(x, M) + self._residual(x, A30, A31, M)
            return x / math.sqrt(2)
        else:
            x = self._residual(x, A30, A31, M)
            return x


class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super(AdaIN, self).__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class ST_AdaINResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim, actv=nn.LeakyReLU(0.2),
                 normalize=False, upsample=False, highpass=True):
        super(ST_AdaINResBlk, self).__init__()
        self.actv = actv
        self.normalize = normalize
        self.upsample = upsample
        self.highpass = highpass
        self.projection = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim):
        self.conv1 = ST_ConvTranspose(dim_in, dim_out, kernel_size=(3, 3), upsample=self.upsample)
        self.conv2 = ST_ConvTranspose(dim_out, dim_out, kernel_size=(3, 3))
        if self.projection:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False)
        if self.normalize:
            self.norm1 = AdaIN(style_dim, dim_in)
            self.norm2 = AdaIN(style_dim, dim_out)
        if self.upsample:
            self.s_unpool = S_AvgUnpool()
            self.t_unpool = T_AvgUnpool()

    def _shortcut(self, x, M):
        if self.upsample:
            x = self.s_unpool(x, M)
            x = self.t_unpool(x)
        if self.projection:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, A30, A31, M, s):
        A3 = A30 if A30 is not None else A31
        if self.normalize:
            x = self.norm1(x, s)
        x = self.actv(x)
        x = self.conv1(x, A3, M)
        if self.upsample:
            A3 = A31

        if self.normalize:
            x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x, A3)
        return x

    def forward(self, x, A30=None, A31=None, M=None, s=None):
        if self.highpass:
            x = self._shortcut(x, M) + self._residual(x, A30, A31, M, s)
            return x / math.sqrt(2)
        else:
            x = self._residual(x, A30, A31, M, s)
            return x


###############################################################################
# Networks
###############################################################################
class Encoder(nn.Module):
    def __init__(self, nef, n_blk, n_btn,
                 normalize=False, downsample=False, highpass=True):
        super(Encoder, self).__init__()
        self.n_blk = n_blk
        self.encode = nn.ModuleList()
        self.bottleneck = nn.ModuleList()

        # downsampling blocks
        for i in range(n_blk):
            mult = 2**i
            self.encode.append(
                ST_ResBlk(nef * mult, nef * mult * 2, normalize=normalize, downsample=downsample, highpass=highpass))

        # bottleneck blocks
        mult = 2 ** n_blk
        for i in range(int(n_btn / 2)):
            self.bottleneck.append(
                ST_ResBlk(nef * mult, nef * mult, normalize=normalize, highpass=highpass))

    def forward(self, x, A):
        i = 1
        for block in self.encode:
            x = block(x,
                      A30=A[i]['A30'],
                      A31=A[i]['A31'],
                      M=A[i]['M'])
            i += 1

        i = min(i, self.n_blk)
        for block in self.bottleneck:
            x = block(x,
                      A31=A[i]['A31'])
        return x


class Decoder(nn.Module):
    def __init__(self, ndf, style_dim, n_blk, n_btn,
                 normalize=False, upsample=False, highpass=True):
        super(Decoder, self).__init__()
        self.n_blk = n_blk
        self.decode = nn.ModuleList()
        self.bottleneck = nn.ModuleList()

        # bottleneck blocks
        mult = 2 ** n_blk
        for i in range(int(n_btn / 2)):
            self.bottleneck.append(
                ST_AdaINResBlk(ndf * mult, ndf * mult, style_dim, normalize=normalize, highpass=highpass)
            )

        # upsampling blocks
        for i in range(n_blk):
            mult = 2**i
            self.decode.insert(
                0, ST_AdaINResBlk(ndf * mult * 2, ndf * mult, style_dim, normalize=normalize, upsample=upsample, highpass=highpass))

    def forward(self, x, A, s):
        i = 1
        for block in self.bottleneck:
            x = block(x,
                      A30=A[i]['A30'],
                      s=s)

        for block in self.decode:
            x = block(x,
                      A30=A[i]['A30'],
                      A31=A[i]['A31'],
                      M=A[i]['M'],
                      s=s)
            i += 1
        return x


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, style_dim=64, n_blk=2, n_btn=0):
        super(Generator, self).__init__()
        self.from_xyz = nn.Conv2d(input_nc, ngf, kernel_size=1)
        self.encoder = Encoder(ngf, n_blk, n_btn, normalize=True, downsample=True, highpass=True)
        self.decoder = Decoder(ngf, style_dim, n_blk, n_btn, normalize=True, upsample=True, highpass=True)
        self.to_xyz = nn.Conv2d(ngf, output_nc, kernel_size=1)

    def forward(self, x, enc_A, dec_A, s, output_z=False):
        x = self.from_xyz(x)
        x = self.encoder(x, enc_A)
        z = x
        x = self.decoder(x, dec_A, s)
        x = self.to_xyz(x)
        if not output_z:
            return x
        else:
            return x, z


class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=16, hidden_dim=512, style_dim=64, num_domains=3):
        super(MappingNetwork, self).__init__()
        self.num_domains = num_domains

        layers = []
        layers += [nn.Linear(latent_dim, hidden_dim // 2)]
        layers += [nn.ReLU()]
        self.fc_latent = nn.Sequential(*layers)

        layers = []
        layers += [nn.Linear(num_domains, hidden_dim // 2)]
        layers += [nn.ReLU()]
        self.fc_label = nn.Sequential(*layers)

        layers = []
        layers += [nn.Linear(hidden_dim, hidden_dim)]
        layers += [nn.ReLU()]
        layers += [nn.Linear(hidden_dim, hidden_dim * 2)]
        layers += [nn.ReLU()]
        layers += [nn.Linear(hidden_dim * 2, style_dim)]
        layers += [nn.Tanh()]
        self.shared = nn.Sequential(*layers)

    def forward(self, z, y):
        label = torch.zeros(y.size(0), self.num_domains).to(y.device)
        idx = range(y.size(0))
        label[idx, y] = 1

        z = self.fc_latent(z)
        y = self.fc_label(label)
        zy = torch.cat([z, y], dim=1)
        s = self.shared(zy)
        return s


class StyleEncoder(nn.Module):
    def __init__(self, input_nc, nef=64, style_dim=64, num_domains=3, clip_size=(240, 21), n_blk=2, n_btn=0):
        super(StyleEncoder, self).__init__()
        layers = []
        layers += [nn.Conv2d(input_nc, nef, kernel_size=1)]
        layers += [Encoder(nef, n_blk, n_btn, downsample=True, highpass=True)]

        mult = 2 ** n_blk
        kw = clip_size[0] // mult
        kh = clip_size[1] // mult
        layers += [nn.LeakyReLU(0.2)]
        layers += [nn.Conv2d(nef * mult, nef * mult, kernel_size=(kw, kh), stride=1, padding=0)]
        layers += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*layers)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Linear(nef * mult, style_dim)]

    def forward(self, x, enc_A, y):
        for layer in self.shared:
            if layer.__class__.__name__ == 'Encoder':
                x = layer(x, enc_A)
            else:
                x = layer(x)

        x = x.view(x.size(0), -1)
        z = x
        out = []
        for branch in self.unshared:
            out += [branch(x)]
        out = torch.stack(out, dim=1)
        idx = range(y.size(0))
        s = out[idx, y]

        return s


class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, num_domains=3, clip_size=(240, 21), n_blk=2, n_btn=0):
        super(Discriminator, self).__init__()
        layers = []
        layers += [nn.Conv2d(input_nc, ndf, kernel_size=1)]
        layers += [Encoder(ndf, n_blk, n_btn, downsample=True, highpass=True)]

        mult = 2 ** n_blk
        kw = clip_size[0] // mult
        kh = clip_size[1] // mult
        layers += [nn.LeakyReLU(0.2)]
        layers += [nn.Conv2d(ndf * mult, ndf * mult, kernel_size=(kw, kh), stride=1, padding=0)]
        layers += [nn.LeakyReLU(0.2)]
        layers += [nn.Conv2d(ndf * mult, num_domains, 1, 1, 0)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x, enc_A, y):
        for layer in self.layers:
            if layer.__class__.__name__ == 'Encoder':
                x = layer(x, enc_A)
            else:
                x = layer(x)

        out = x.view(x.size(0), -1)
        idx = range(y.size(0))
        out = out[idx, y]
        return out
import os
import random

import numpy as np
import torch
from networks.networks import *
from networks.blocks import *

import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import tensorflow as tf
from collections import OrderedDict
from os.path import join as pjoin
import codecs as cs
from utils.utils import *
import time

class Logger:
    def __init__(self, log_dir):
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()

class BaseTrainer:
    @staticmethod
    def zero_grad(opt_list):
        for opt in opt_list:
            opt.zero_grad()

    @staticmethod
    def clip_norm(network_list):
        for network in network_list:
            clip_grad_norm_(network.parameters(), 0.5)

    @staticmethod
    def step(opt_list):
        for opt in opt_list:
            opt.step()

    @staticmethod
    def to(net_opt_list, device):
        for net_opt in net_opt_list:
            net_opt.to(device)

    @staticmethod
    def net_train(network_list):
        for network in network_list:
            network.train()

    @staticmethod
    def net_eval(network_list):
        for network in network_list:
            network.eval()

    @staticmethod
    def swap(x):
        "Swaps the ordering of the minibatch"
        shape = x.shape
        assert shape[0] % 2 == 0, "Minibatch size must be a multiple of 2"
        new_shape = [shape[0]//2, 2] + list(shape[1:])
        x = x.view(*new_shape)
        x = torch.flip(x, [1])
        return x.view(*shape)

    @staticmethod
    def grid_sample_1d(x, target_size, scale_range, num_crops=1):
        # build grid
        B = x.size(0) * num_crops
        unit_grid = torch.linspace(-1.0, 1.0, target_size, device=x.device).view(1, -1, 1, 1).expand(B, -1, -1, -1)
        #   (B, target_size, 1, 2)
        unit_grid = torch.cat([torch.ones_like(unit_grid) * -1, unit_grid], dim=3)

        # print(x.shape)
        #   (B // num_crops, D, Seq_len) -> (B, D, Seq_len, 1)
        x = x.unsqueeze(1).unsqueeze(-1).expand(-1, num_crops, -1, -1, -1).flatten(0, 1)

        scale = torch.rand(B, 1, 1, 1, device=x.device) * (scale_range[1] - scale_range[0]) + scale_range[0]
        offset = (torch.rand(B, 1, 1, 1, device=x.device) * 2 - 1) * (1 - scale)
        sampling_grid = unit_grid * scale + offset
        sampling_grid[:, :, :, 0] = -1
        #   (B, D, target_size, 1)
        crop = F.grid_sample(x, sampling_grid, align_corners=False, padding_mode="border")
        #   (B, D, target_size, Seq_len)
        crop = crop.view(B // num_crops, num_crops, crop.size(1), crop.size(2))
        return crop

    # @staticmethod
    def get_random_crops(self, data, grid_sample=True):
        if grid_sample:
            return self.grid_sample_1d(data, self.opt.patch_size,
                                       (self.opt.patch_min_scale, self.opt.patch_max_scale),
                                       self.opt.num_crops)
        # scale = self.opt.patch_size
        B, D, L = data.shape
        data = data.unsqueeze(1).expand(-1, self.opt.num_crops, -1, -1).flatten(0, 1)
        start_idx = np.random.randint(0, L-self.opt.patch_size-1, data.shape[0])
        res_data = []
        for i in range(data.size(0)):
            res_data.append(data[i:i+1, :, start_idx[i]:start_idx[i]+self.opt.patch_size])
        data = torch.cat(res_data, dim=0)
        return data.view(B, -1, D, self.opt.patch_size)

class LatentVAETrainer(BaseTrainer):
    def __init__(self, opt, encoder, decoder, ae_encoder, ae_decoder):
        self.opt = opt
        self.encoder = encoder
        self.decoder = decoder
        self.ae_encoder = ae_encoder
        self.ae_decoder = ae_decoder

        if self.opt.is_train:
            self.logger = Logger(self.opt.log_dir)
            self.mse_criterion = torch.nn.MSELoss()
            self.l1_criterion = torch.nn.SmoothL1Loss()

    @staticmethod
    def reparametrize(mu, logvar):
        s_var = logvar.mul(0.5).exp_()
        eps = s_var.data.new(s_var.size()).normal_()
        return eps.mul(s_var).add_(mu)

    @staticmethod
    def ones_like(tensor, val=1.):
        return torch.FloatTensor(tensor.size()).fill_(val).to(tensor.device).requires_grad_(False)

    @staticmethod
    def zeros_like(tensor, val=0.):
        return torch.FloatTensor(tensor.size()).fill_(val).to(tensor.device).requires_grad_(False)

    @staticmethod
    def kl_criterion(mu1, logvar1, mu2, logvar2):

        sigma1 = logvar1.mul(0.5).exp()
        sigma2 = logvar2.mul(0.5).exp()
        kld = torch.log(sigma2 / sigma1) + (torch.exp(logvar1) + (mu1 - mu2) ** 2) / (
                2 * torch.exp(logvar2)) - 1 / 2
        return kld.sum() / np.prod(mu1.shape)

    @staticmethod
    def kl_criterion_unit(mu, logvar):
        kld = ((torch.exp(logvar) + mu ** 2) - logvar - 1) / 2
        return kld.sum() / np.prod(mu.shape)

    def forward(self, batch_data):
        if self.opt.dataset_name == "cmu":
            M1, M2, A1, S1, SID1 = batch_data
        else:
            M1, M2, MS, _, A1, S1, SID1, _, _ = batch_data
        A2, S2 = A1, S1

        M1 = M1.permute(0, 2, 1).to(self.opt.device).float().detach()
        M2 = M2.permute(0, 2, 1).to(self.opt.device).float().detach()
        M3 = self.swap(M2)
        SID3 = self.swap(SID1)

        if self.opt.use_action:
            A1 = A1.to(self.opt.device).float().detach()
            A2 = A2.to(self.opt.device).float().detach()
            A3 = self.swap(A2)
        else:
            A1, A2, A3 = None, None, None

        if self.opt.use_style:
            S1 = S1.to(self.opt.device).float().detach()
            S2 = S2.to(self.opt.device).float().detach()
            S3 = self.swap(S2)
        else:
            S1, S2, S3 = None, None, None

        LM1, _, _ = self.ae_encoder(M1[:, :-4])
        LM2, _, _ = self.ae_encoder(M2[:, :-4])
        LM3 = self.swap(LM2)

        LM1, LM2, LM3 = LM1.detach(), LM2.detach(), LM3.detach()

        sp1, gl_mu1, gl_logvar1 = self.encoder(LM1, A1, S1)
        sp2, gl_mu2, gl_logvar2 = self.encoder(LM2, A2, S2)
        sp3, gl_mu3, gl_logvar3 = self.swap(sp2), self.swap(gl_mu2), self.swap(gl_logvar2)

        z_sp1 = sp1
        z_gl1 = self.reparametrize(gl_mu1, gl_logvar1)

        z_sp2 = sp2
        # May detach the graph of M1
        z_gl2 = self.reparametrize(gl_mu2, gl_logvar2)

        # May detach the graph of M2
        z_sp3 = z_sp2.detach()
        z_gl3 = self.reparametrize(gl_mu3, gl_logvar3)


        RLM1 = self.decoder(z_sp1, z_gl1, A1, S1)
        RLM2 = self.decoder(z_sp2, z_gl2, A2, S2)
        RLM3 = self.decoder(z_sp3, z_gl3, A2, S3)

        # Should be identical to M2
        # May detach from graph of RM3
        sp4, gl_mu4, gl_logvar4 = self.encoder(RLM3, A2, S3)


        z_sp4 = sp4
        # May detach from graph of M2
        z_gl4 = self.reparametrize(gl_mu2.detach(), gl_logvar2.detach())

        #  Should be identical to M3
        # May detach from graph of M3
        z_sp5 = sp3.detach()
        z_gl5 = self.reparametrize(gl_mu4, gl_logvar4)


        RRLM2 = self.decoder(z_sp4, z_gl4, A2, S2)
        RRLM3 = self.decoder(z_sp5, z_gl5, A3, S3)

        RM1 = self.ae_decoder(RLM1)
        RM2 = self.ae_decoder(RLM2)
        RM3 = self.ae_decoder(RLM3)
        RRM2 = self.ae_decoder(RRLM2)
        RRM3 = self.ae_decoder(RRLM3)

        self.M1, self.M2, self.M3 = M1, M2, M3
        self.LM1, self.LM2, self.LM3 = LM1, LM2, LM3
        self.RLM1, self.RLM2 = RLM1, RLM2
        self.RRLM2, self.RRLM3 = RRLM2, RRLM3
        self.RM1, self.RM2, self.RM3, self.RRM2, self.RRM3 = RM1, RM2, RM3, RRM2, RRM3
        self.SID1 = SID1
        self.SID3 = SID3
        self.sp2, self.sp4 = sp2, sp4
        self.gl_mu1, self.gl_mu2, self.gl_mu3, self.gl_mu4 = gl_mu1, gl_mu2, gl_mu3, gl_mu4
        self.gl_logvar1, self.gl_logvar2, self.gl_logvar3, self.gl_logvar4 = gl_logvar1, gl_logvar2, gl_logvar3, gl_logvar4

    def generate(self, M1, M2, S2, sampling):
        M1 = M1.clone()
        M2 = M2.clone()

        M1 = M1.permute(0, 2, 1).to(self.opt.device).float().detach()
        M2 = M2.permute(0, 2, 1).to(self.opt.device).float().detach()

        if self.opt.use_style:
            S2 = S2.to(self.opt.device).float().detach()
        else:
            S2 = None
        LM1, _, _ = self.ae_encoder(M1[:, :-4])
        LM2, _, _ = self.ae_encoder(M2[:, :-4])

        sp1 = self.encoder.extract_content_feature(LM1, None)
        gl_mu2, gl_logvar2 = self.encoder.extract_style_feature(LM2, S2)

        z_sp = sp1
        if sampling:
            # Sample from normal distribution, novel style generation
            z_gl = self.reparametrize(self.zeros_like(gl_mu2), self.zeros_like(gl_logvar2))
        else:
            # Sample from M2 distribution, motion style transfer
            z_gl = self.reparametrize(gl_mu2, gl_logvar2)

        TLM = self.decoder(z_sp, z_gl, None, S2)
        TM = self.ae_decoder(TLM)

        return TM.permute(0, 2, 1)


    def backward(self):
        self.loss_rec_lm1 = self.l1_criterion(self.LM1, self.RLM1)
        self.loss_rec_lm2 = self.l1_criterion(self.LM2, self.RLM2)
        self.loss_rec_m1 = self.l1_criterion(self.M1, self.RM1)
        self.loss_rec_m2 = self.l1_criterion(self.M2, self.RM2)

        self.loss_rec_rlm2 = self.l1_criterion(self.LM2, self.RRLM2)
        self.loss_rec_rlm3 = self.l1_criterion(self.LM3, self.RRLM3)
        self.loss_rec_rm2 = self.l1_criterion(self.M2, self.RRM2)
        self.loss_rec_rm3 = self.l1_criterion(self.M3, self.RRM3)

        self.loss_rec_lat = self.l1_criterion(self.sp2, self.sp4)

        self.loss_kld_gl_m1 = self.kl_criterion_unit(self.gl_mu1, self.gl_logvar1)
        self.loss_kld_gl_m2 = self.kl_criterion_unit(self.gl_mu2, self.gl_logvar2)
        self.loss_kld_gl_m4 = self.kl_criterion_unit(self.gl_mu4, self.gl_logvar4)
        self.loss_kld_gl_m12 = self.kl_criterion(self.gl_mu1, self.gl_logvar1, self.gl_mu2, self.gl_logvar2)
        self.loss_kld_gl_m34 = self.kl_criterion(self.gl_mu3, self.gl_logvar3, self.gl_mu4, self.gl_logvar4)

        self.loss = (self.loss_rec_lm1 + self.loss_rec_lm2 + self.loss_rec_m1 + self.loss_rec_m2) * self.opt.lambda_rec + \
                    (self.loss_rec_rlm2 + self.loss_rec_rlm3 + self.loss_rec_rm2 + self.loss_rec_rm3) * self.opt.lambda_rec_c + \
                    (self.loss_kld_gl_m1 + self.loss_kld_gl_m2 + self.loss_kld_gl_m4) * self.opt.lambda_kld_gl + \
                    self.loss_kld_gl_m12 * self.opt.lambda_kld_gl12


        loss_logs = OrderedDict({})
        loss_logs["loss"] = self.loss.item()
        loss_logs["loss_rec_lm1"] = self.loss_rec_lm1.item()
        loss_logs["loss_rec_lm2"] = self.loss_rec_lm2.item()
        loss_logs["loss_rec_m1"] = self.loss_rec_m1.item()
        loss_logs["loss_rec_m2"] = self.loss_rec_m2.item()

        loss_logs["loss_rec_rlm2"] = self.loss_rec_rlm2.item()
        loss_logs["loss_rec_rlm3"] = self.loss_rec_rlm3.item()
        loss_logs["loss_rec_rm2"] = self.loss_rec_rm2.item()
        loss_logs["loss_rec_rm3"] = self.loss_rec_rm3.item()
        loss_logs["loss_rec_lat"] = self.loss_rec_lat.item()

        loss_logs["loss_kld_gl_m1"] = self.loss_kld_gl_m1.item()
        loss_logs["loss_kld_gl_m2"] = self.loss_kld_gl_m2.item()
        loss_logs["loss_kld_gl_m4"] = self.loss_kld_gl_m4.item()
        loss_logs["loss_kld_gl_m12"] = self.loss_kld_gl_m12.item()
        loss_logs["loss_kld_gl_m34"] = self.loss_kld_gl_m34.item()

        return loss_logs

    def update(self):
        self.zero_grad([self.opt_encoder, self.opt_decoder])
        loss_logs = self.backward()
        self.loss.backward()
        self.clip_norm([self.encoder, self.decoder])
        self.step([self.opt_encoder, self.opt_decoder])
        return loss_logs

    def save(self, file_name, ep, total_it):
        state = {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "ae_encoder": self.ae_encoder.state_dict(),
            "ae_decoder": self.ae_decoder.state_dict(),

            "opt_encoder": self.opt_encoder.state_dict(),
            "opt_decoder": self.opt_decoder.state_dict(),
            "opt_ae_decoder": self.opt_ae_decoder.state_dict(),

            "ep": ep,
            "total_it": total_it,
        }

        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.opt.device)
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.decoder.load_state_dict(checkpoint["decoder"])
        self.ae_encoder.load_state_dict(checkpoint["ae_encoder"])
        self.ae_decoder.load_state_dict(checkpoint["ae_decoder"])

        if self.opt.is_train:
            self.opt_encoder.load_state_dict(checkpoint["opt_encoder"])
            self.opt_decoder.load_state_dict(checkpoint["opt_decoder"])
            self.opt_ae_decoder.load_state_dict(checkpoint["opt_ae_decoder"])
        print("Loading the model from epoch %04d"%checkpoint["ep"])
        return checkpoint["ep"], checkpoint["total_it"]

    def train(self, train_dataloader, val_dataloader, plot_eval):
        net_list = [self.encoder, self.decoder, self.ae_encoder, self.ae_decoder]
        self.to(net_list, self.opt.device)

        self.opt_encoder = optim.Adam(self.encoder.parameters(), lr=self.opt.lr)
        self.opt_decoder = optim.Adam(self.decoder.parameters(), lr=self.opt.lr)
        self.opt_ae_decoder = optim.Adam(self.decoder.parameters(), lr=self.opt.lr*0.1)

        epoch = 0
        it = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, "latest.tar")
            epoch, it = self.resume(model_dir)
            print("Loading model from Epoch %d" % (epoch))

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_dataloader)
        print("Iters Per Epoch, Training: %04d, Validation: %03d" % (len(train_dataloader), len(val_dataloader)))
        min_val_loss = np.inf
        logs = OrderedDict()

        while epoch < self.opt.max_epoch:
            self.net_train(net_list)
            for i, batch_data in enumerate(train_dataloader):
                self.forward(batch_data)
                loss_dict = self.update()

                for k, v in loss_dict.items():
                    if k not in logs:
                        logs[k] = v
                    else:
                        logs[k] += v
                it += 1
                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()

                    for tag, value in logs.items():
                        self.logger.scalar_summary(tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = OrderedDict()
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch, i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)
            epoch += 1

            print("Validation time:")
            val_loss = None
            with torch.no_grad():
                self.net_eval(net_list)
                for i, batch_data in enumerate(val_dataloader):
                    self.forward(batch_data)
                    loss_dict = self.backward()
                    if val_loss is None:
                        val_loss = loss_dict
                    else:
                        for k, v in loss_dict.items():
                            val_loss[k] += v

            print_str = "Validation Loss:"
            for k, v in val_loss.items():
                val_loss[k] /= len(val_dataloader)

                print_str += ' %s: %.4f ' % (k, val_loss[k])
            self.logger.scalar_summary("val_loss", val_loss["loss"], epoch)
            print(print_str)

            if val_loss["loss"] < min_val_loss:
                min_val_loss = val_loss["loss"]
                min_val_epoch = epoch
                self.save(pjoin(self.opt.model_dir, "best.tar"), epoch, it)
                print("Best Validation Model So Far!~")

            if epoch % self.opt.eval_every_e == 0:
                B = self.M1.size(0)
                data = torch.cat([self.M2[:6:2], self.RM2[:6:2], self.M3[:6:2], self.RM3[:6:2]],
                                 dim=0)
                styles = torch.cat([self.SID1[:6:2], self.SID1[:6:2], self.SID3[:6:2], self.SID3[:6:2]],
                                   dim=0).detach().cpu().numpy()
                data = data.permute(0, 2, 1).detach().cpu().numpy()
                save_dir = pjoin(self.opt.eval_dir, "E%04d" % (epoch))
                os.makedirs(save_dir, exist_ok=True)
                plot_eval(data, save_dir, styles)

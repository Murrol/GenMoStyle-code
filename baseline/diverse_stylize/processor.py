import numpy as np
import torch
import torch.nn.functional as F
from data.skeleton_graph import SkeletonGraph
from collections import OrderedDict
import os

parents = [-1,
            0, 1, 2, 3,
            0, 5, 6, 7,
            0, 9, 10, 11,
            10, 13, 14, 15,
            10, 17, 18, 19]


class Processor(object):
    def __init__(self, opt):
        self.opt = opt
        self.joint_num = opt.clip_size[-1]
        self.device = torch.device("cuda:{}".format(opt.gpu_ids[0]) if torch.cuda.is_available() else "cpu")
        self.matrix_names = ['A10', 'A11', 'A30', 'A31', 'M']
        # preprocess = np.load(opt.preproot)
        # self.Xmean = torch.from_numpy(preprocess['Xmean'])[None, :].repeat(opt.batch_size, 1, 1, 1).to(self.device, dtype=torch.float)
        # self.Xstd = torch.from_numpy(preprocess['Xstd'])[None, :].repeat(opt.batch_size, 1, 1, 1).to(self.device, dtype=torch.float)
        data_norm_dir = "../../motion_transfer_data/processed_bfa"
        motion_mean_path = os.path.join(data_norm_dir, "Mean.npy")
        motion_std_path = os.path.join(data_norm_dir, "Std.npy")

        if os.path.exists(motion_mean_path) and os.path.exists(motion_std_path):
            self.motion_mean = np.load(motion_mean_path, allow_pickle=True).astype(np.float32)
            self.motion_std = np.load(motion_std_path, allow_pickle=True).astype(np.float32)

        else:
            assert self.motion_mean and self.motion_std, 'no motion_mean or no motion_std'

        root_mean = np.zeros_like(self.motion_mean[: 4])[None, None].repeat(self.joint_num, axis=1)
        pos_mean = self.motion_mean[4: 4 + self.joint_num * 3].reshape(-1, self.joint_num, 3)
        rot_mean = self.motion_mean[4 + self.joint_num * 3 : 4 + self.joint_num * 9].reshape(-1, self.joint_num, 6)
        root_std = np.ones_like(self.motion_std[: 4])[None, None].repeat(self.joint_num, axis=1)
        pos_std = self.motion_std[4: 4 + self.joint_num * 3].reshape(-1, self.joint_num, 3)
        rot_std = self.motion_std[4 + self.joint_num * 3 : 4 + self.joint_num * 9].reshape(-1, self.joint_num, 6)
        self.clips_mean = np.concatenate((pos_mean, rot_mean, root_mean), axis=-1)
        self.clips_std = np.concatenate((pos_std, rot_std, root_std), axis=-1)
        self.clips_mean = np.transpose(self.clips_mean, (2, 0, 1))
        self.clips_std = np.transpose(self.clips_std, (2, 0, 1))
        self.Xmean = torch.from_numpy(self.clips_mean)[None, :].repeat(opt.batch_size, 1, 1, 1).to(self.device, dtype=torch.float)
        self.Xstd = torch.from_numpy(self.clips_std)[None, :].repeat(opt.batch_size, 1, 1, 1).to(self.device, dtype=torch.float)

        # set adjacency matrices
        self.set_adjacency(sampling_blk=opt.ng_blk, encoding_blk=opt.ne_blk)

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def set_adjacency(self, sampling_blk, encoding_blk):
        graph = SkeletonGraph()

        down_A = OrderedDict()
        for i in range(sampling_blk):
            mats = graph.get_adjacency('encode', i + 1)
            dic = OrderedDict()
            for j, mat in enumerate(self.matrix_names):
                dic[mat] = mats[j].to(self.device, dtype=torch.float)
            down_A[i + 1] = dic

        up_A = OrderedDict()
        for i in range(sampling_blk):
            mats = graph.get_adjacency('decode', (sampling_blk - i))
            dic = OrderedDict()
            for j, mat in enumerate(self.matrix_names):
                dic[mat] = mats[j].to(self.device, dtype=torch.float)
            up_A[i + 1] = dic

        enc_A = OrderedDict()
        for i in range(encoding_blk):
            mats = graph.get_adjacency('encode', i + 1)
            dic = OrderedDict()
            for j, mat in enumerate(self.matrix_names):
                dic[mat] = mats[j].to(self.device, dtype=torch.float)
            enc_A[i + 1] = dic

        self.down_A = down_A
        self.up_A = up_A
        self.enc_A = enc_A

    def compute_d_loss(self, model, inputs, alter='latent'):
        x_real = inputs['x_real']['posrot']
        y_org = inputs['y_org']
        # print(x_real.shape)
        if alter == 'latent':
            z_trg = inputs['z_trg']
        elif alter == 'ref':
            x_ref = inputs['x_ref']['posrot']
        else:
            raise NotImplementedError()
        y_trg = inputs['y_trg']

        # with real ones
        x_real.requires_grad_()
        out = model.netD(x_real, self.enc_A, y_org)
        loss_real = adv_loss(out, 1)
        loss_reg = r1_reg(out, x_real)

        # with fake ones
        with torch.no_grad():
            if alter == 'latent':
                s_trg = model.netF(z_trg, y_trg)
            elif alter == 'ref':
                s_trg = model.netE(x_ref, self.enc_A, y_trg)
            else:
                raise NotImplementedError()
            x_fake = model.netG(x_real, self.down_A, self.up_A, s_trg)
        out = model.netD(x_fake, self.enc_A, y_trg)
        loss_fake = adv_loss(out, 0)

        loss = loss_real + loss_fake + self.opt.lambda_reg * loss_reg
        losses = OrderedDict([('loss', loss.item()),
                              ('real', loss_real.item()),
                              ('fake', loss_fake.item()),
                              ('reg', loss_reg.item())])

        return loss, losses

    def compute_g_loss(self, model, inputs, alter='latent'):
        x_real = inputs['x_real']['posrot']
        x_real_traj = inputs['x_real']['traj']
        x_real_feet = inputs['x_real']['feet']
        y_org = inputs['y_org']

        if alter == 'latent':
            z_trg = inputs['z_trg']
            z_trg2 = inputs['z_trg2']
        elif alter == 'ref':
            x_ref = inputs['x_ref']['posrot']
            x_ref2 = inputs['x_ref2']['posrot']
        else:
            raise NotImplementedError()
        y_trg = inputs['y_trg']

        # adversarial loss
        if alter == 'latent':
            s_trg = model.netF(z_trg, y_trg)
        elif alter == 'ref':
            s_trg = model.netE(x_ref, self.enc_A, y_trg)
        else:
            raise NotImplementedError()
        x_fake = model.netG(x_real, self.down_A, self.up_A, s_trg)
        out = model.netD(x_fake, self.enc_A, y_trg)
        loss_adv = adv_loss(out, 1)

        # content reconstruction loss
        s_org = model.netE(x_real, self.enc_A, y_org)
        x_rec = model.netG(x_real, self.down_A, self.up_A, s_org)
        loss_con = torch.mean((x_rec - x_real).norm(dim=3))

        # style reconstruction loss
        s_pred = model.netE(x_fake, self.enc_A, y_trg)
        loss_sty = torch.mean(torch.abs(s_pred - s_trg))

        if alter == 'latent':
            s_trg2 = model.netF(z_trg2, y_trg)
        elif alter == 'ref':
            s_trg2 = model.netE(x_ref2, self.enc_A, y_trg)
        else:
            raise NotImplementedError()

        # diversity sensitive loss
        x_fake2 = model.netG(x_real, self.down_A, self.up_A, s_trg2)
        x_fake2 = x_fake2.detach()
        loss_ds = torch.mean(torch.abs(x_fake - x_fake2))

        # cycle-consistency loss
        x_cyc = model.netG(x_fake, self.down_A, self.up_A, s_org)
        loss_cyc = torch.mean((x_cyc - x_real).norm(dim=3))

        if alter == 'latent':
            output = {'x_fake_latent': x_fake,
                      'x_fake_latent2': x_fake2}
        elif alter == 'ref':
            output = {'x_fake_ref': x_fake,
                      'x_fake_ref2': x_fake2}
        else:
            raise NotImplementedError()

        loss = self.opt.lambda_adv * loss_adv \
               + self.opt.lambda_sty * loss_sty \
               + self.opt.lambda_cyc * loss_cyc \
               + self.opt.lambda_con * loss_con \
               - self.opt.lambda_ds * loss_ds \

        losses = OrderedDict([('loss', loss.item()),
                              ('adv', loss_adv.item()),
                              ('con', loss_con.item()),
                              ('sty', loss_sty.item()),
                              ('ds', loss_ds.item()),
                              ('cyc', loss_cyc.item())])

        return loss, losses, output

    def test(self,  model, inputs, alter='latent'):
        model.eval()
        x_real = inputs['x_real']['posrot']
        x_real_traj = inputs['x_real']['traj']
        y_org = inputs['y_org']

        if alter == 'latent':
            z_trg = inputs['z_trg']
        elif alter == 'ref':
            x_ref = inputs['x_ref']['posrot']
        else:
            raise NotImplementedError()
        y_trg = inputs['y_trg']

        with torch.no_grad():
            if alter=='latent':
                s_trg = model.netF(z_trg, y_trg)
                numparam = self.count_parameters(model.netF) + self.count_parameters(model.netG)
            elif alter == 'ref':
                s_trg = model.netE(x_ref, self.enc_A, y_trg)
                numparam = self.count_parameters(model.netE) + self.count_parameters(model.netG)
            else:
                raise NotImplementedError()

            x_fake = model.netG(x_real, self.down_A, self.up_A, s_trg)
        print(f"model num parameter {numparam:.6e}")
        return x_fake


def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)

    return loss


def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    
    return reg

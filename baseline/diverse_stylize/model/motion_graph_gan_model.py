import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import st_networks


class MotionGraphGANModel(nn.Module):
    def __init__(self, opt):
        super(MotionGraphGANModel, self).__init__()
        self.opt = opt
        self.mode = opt.mode
        self.device = torch.device("cuda:{}".format(opt.gpu_ids[0]) if torch.cuda.is_available() else "cpu")
        self.save_dir = opt.save_dir
        self.model_names = ['G', 'E', 'F', 'D']

        # define networks
        self.netG = st_networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.style_dim, opt.ng_blk, opt.ng_btn)
        self.netE = st_networks.define_E(opt.input_nc, opt.nef, opt.style_dim, opt.num_domains, opt.clip_size, opt.ne_blk, opt.ne_btn)
        self.netF = st_networks.define_F(opt.latent_dim, opt.hidden_dim, opt.style_dim, opt.num_domains)
        self.netD = st_networks.define_D(opt.input_nc, opt.ndf, opt.num_domains, opt.clip_size, opt.nd_blk, opt.nd_btn)

        # set optimizers
        if self.mode == 'train':
            for name in self.model_names:
                setattr(self, 'optimizer_' + name, self.set_optimizer(name))

        # set lr schedulers
        if self.mode == 'train':
            for name in self.model_names:
                setattr(self, 'scheduler_' + name, self.set_scheduler(name))

        self.to(self.device)

    def set_optimizer(self, name):
        net = getattr(self, 'net' + name)
        if name == 'F':
            lr = self.opt.f_lr
        elif name == 'G':
            lr = self.opt.g_lr
        elif name == 'D':
            lr = self.opt.d_lr
        elif name == 'E':
            lr = self.opt.e_lr
        else:
            NotImplementedError()

        optimizer = torch.optim.Adam(
            params=net.parameters(),
            lr=lr,
            betas=(self.opt.beta1, self.opt.beta2),
            weight_decay=self.opt.weight_decay)
        return optimizer

    def set_scheduler(self, name):
        optimizer = getattr(self, 'optimizer_' + name)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=0.95
        )
        return scheduler

    def reset_grad(self):
        for name in self.model_names:
            optim = getattr(self, 'optimizer_' + name)
            optim.zero_grad()

    def get_current_iter(self):
        return self.current_iter

    def get_current_lrs(self):
        learning_rates = {}
        for name in self.model_names:
            optimizer = getattr(self, 'optimizer_' + name)
            for param_group in optimizer.param_groups:
                learning_rates[name] = param_group['lr']
        return learning_rates

    def print_networks(self):
        for name in self.model_names:
            save_path = os.path.join(self.save_dir, 'net%s.txt' % name)
            with open(save_path, 'w') as nets_f:
                if isinstance(name, str):
                    net = getattr(self, 'net' + name)
                st_networks.print_network(net, nets_f)

    def save_networks(self, iter=None, latest=False):
        if latest:
            save_filename = 'latest_checkpoint.pth'
        else:
            save_filename = '%d_checkpoint.pth' % iter
        save_path = os.path.join(self.save_dir, save_filename)
        print('Saving the model into %s...' % save_path)

        checkpoint = {'iter': iter}
        for name in self.model_names:
            if isinstance(name, str):
                net_name = 'net' + name
                optim_name = 'optimizer_' + name
                net = getattr(self, net_name)
                optim = getattr(self, optim_name)
                checkpoint[net_name + '_state_dict'] = net.state_dict()
                checkpoint[optim_name + '_state_dict'] = optim.state_dict()
        torch.save(checkpoint, save_path)

    def load_networks(self, iter=None):
        if iter is not None:
            load_filename = '%d_checkpoint.pth' % iter
        else:
            load_filename = 'latest_checkpoint.pth'
        load_path = os.path.join(self.save_dir, load_filename)
        print('Loading the model from %s...' % load_path)

        checkpoint = torch.load(load_path, map_location='cuda:0')
        for name in self.model_names:
            if isinstance(name, str):
                net_name = 'net' + name
                net = getattr(self, net_name)
                net.load_state_dict(checkpoint[net_name + '_state_dict'])

                if self.mode == 'train':
                    optim_name = 'optimizer_' + name
                    optim = getattr(self, optim_name)
                    if name == 'F':
                        lr = self.opt.f_lr
                    elif name == 'G':
                        lr = self.opt.g_lr
                    elif name == 'D':
                        lr = self.opt.d_lr
                    elif name == 'E':
                        lr = self.opt.e_lr
                    optim.load_state_dict(checkpoint[optim_name + '_state_dict'])
                    for param_group in optim.param_groups:
                        param_group['lr'] = lr
        
        self.current_iter = checkpoint['iter']

import sys
import numpy as np
import random
import torch
from torch.utils import data
import os

sys.path.append('../')
from diverse_utils.helper import normalize
# from preprocess.export_dataset import generate_data


f = open('contents_xia.txt', 'r')
contents = [line.strip() for line in f.readlines()]

f = open('styles_bfa.txt', 'r')
styles = [line.strip() for line in f.readlines()]


class SourceDataset(data.Dataset):
    def __init__(self, opt, type='train'):
        self.joint_num = 21
        self.opt = opt
        
        if type == 'train':
            data_npy_path = os.path.join(opt.dataroot, 'train_data.npy')
        elif type == 'test':
            data_npy_path = os.path.join(opt.dataroot, 'test_data.npy')
        
        mdataset = np.load(data_npy_path, allow_pickle=True).item()
        self.motions, self.labels, self.actions = [], [], []
        for key, value in mdataset.items():
            # print(len(value), self.config['motion_length'])
            if len(value)>=opt.motion_length:
                # print(len(value))
                self.motions.append(value)
                if "cmu" in opt.dataroot:
                    self.labels.append(0)
                else:
                    self.labels.append(eval(key.split("#")[-1]))
                if "xia" in opt.dataroot:
                    self.actions.append(eval(key.split("#")[-2]))
                else:
                    self.actions.append(0)

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
        # print(pos_mean.shape, rot_mean.shape, root_mean.shape)
        self.clips_mean = np.concatenate((pos_mean, rot_mean, root_mean), axis=-1).reshape(-1 , self.joint_num, 3+6+4)
        self.clips_std = np.concatenate((pos_std, rot_std, root_std), axis=-1).reshape(-1 , self.joint_num, 3+6+4)
        self.clips_mean = np.transpose(self.clips_mean, (2, 0, 1)) # (C, F, J)
        self.clips_std = np.transpose(self.clips_std, (2, 0, 1))

        self.domains = opt.domains

        # self.clips = np.load(dataroot)['clips']
        # self.feet = np.load(dataroot, allow_pickle=True)['feet']
        self.clips = []
        for motion in self.motions:
            root_data = motion[:, :4][:, None].repeat(self.joint_num, axis=1)
            positions = motion[:, 4:4 + self.joint_num * 3].reshape(-1 , self.joint_num, 3)
            rotations = motion[:, 4 + self.joint_num * 3 : 4 + self.joint_num * 9].reshape(-1 , self.joint_num, 6)
            self.clips.append(np.transpose(np.concatenate((positions, rotations, root_data), axis=-1), (2, 0, 1)))
        self.feet = [motion[:, 4 + self.joint_num * 12:] for motion in self.motions]
        # self.classes = list(zip(self.labels, self.actions))
        self.classes = list(zip(self.actions, self.labels))

        # self.preprocess = np.load(opt.preproot)
        self.samples, self.contacts, self.targets, self.content_labels = self.make_dataset(opt)

    def make_dataset(self, opt):
        X, F, Y, C = [], [], [], []
        for dom in range(opt.num_domains): #bfa classese
            dom_idx = [si for si in range(len(self.classes))
                       if self.classes[si][1] == styles.index(opt.domains[dom])]  # index list that belongs to the domain
            dom_clips = [self.clips[cli] for cli in dom_idx]  # clips list (motion data) that belongs to the domain
            dom_feet = [self.feet[fti] for fti in dom_idx]
            dom_contents = [self.classes[ci][0] for ci in dom_idx]
            X += dom_clips
            F += dom_feet
            Y += [dom] * len(dom_clips)
            C += dom_contents
        return X, F, Y, C

    def __getitem__(self, index):
        x = self.samples[index]
        f = self.contacts[index]
        if self.opt.motion_length > 20:
            roundlen = self.opt.motion_length
        else:
            roundlen = x.shape[1] // 16 * 16
        idx = random.randint(0, x.shape[1] - roundlen)
        x = x[:, idx:idx+roundlen]
        f = f[idx:idx+roundlen]
        x = normalize(x, self.clips_mean, self.clips_std)
        data = {'posrot': x[:9], 'traj': x[-4:], 'feet': f}
        y = self.targets[index]
        c = self.content_labels[index]
        return {'x': data, 'y': y, 'c': c} #y: style, c: content

    def __len__(self):
        return len(self.targets)


class ReferenceDataset(data.Dataset):
    def __init__(self, opt, type='train'):
        self.joint_num = 21
        self.opt = opt
        if type == 'train':
            data_npy_path = os.path.join(opt.dataroot, 'train_data.npy')
        elif type == 'test':
            data_npy_path = os.path.join(opt.dataroot, 'test_data.npy')
        
        mdataset = np.load(data_npy_path, allow_pickle=True).item()
        self.motions, self.labels, self.actions = [], [], []
        for key, value in mdataset.items():
            # print(len(value), self.config['motion_length'])
            if len(value)>=opt.motion_length:
                # print(len(value))
                self.motions.append(value)
                if "cmu" in opt.dataroot:
                    self.labels.append(0)
                else:
                    self.labels.append(eval(key.split("#")[-1]))
                if "xia" in opt.dataroot:
                    self.actions.append(eval(key.split("#")[-2]))
                else:
                    self.actions.append(0)

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
        self.clips_mean = np.concatenate((pos_mean, rot_mean, root_mean), axis=-1).reshape(-1 , self.joint_num, 3+6+4)
        self.clips_std = np.concatenate((pos_std, rot_std, root_std), axis=-1).reshape(-1 , self.joint_num, 3+6+4)
        self.clips_mean = np.transpose(self.clips_mean, (2, 0, 1))
        self.clips_std = np.transpose(self.clips_std, (2, 0, 1))

        self.domains = opt.domains

        # self.clips = np.load(dataroot)['clips']
        # self.feet = np.load(dataroot, allow_pickle=True)['feet']
        self.clips = []
        for motion in self.motions:
            root_data = motion[:, :4][:, None].repeat(self.joint_num, axis=1)
            positions = motion[:, 4:4 + self.joint_num * 3].reshape(-1 , self.joint_num, 3)
            rotations = motion[:, 4 + self.joint_num * 3 : 4 + self.joint_num * 9].reshape(-1 , self.joint_num, 6)
            self.clips.append(np.transpose(np.concatenate((positions, rotations, root_data), axis=-1), (2, 0, 1)))
        self.feet = [motion[:, 4 + self.joint_num * 12:] for motion in self.motions]
        # self.classes = list(zip(self.labels, self.actions))
        self.classes = list(zip(self.actions.copy(), self.labels.copy()))

        # self.preprocess = np.load(opt.preproot)
        self.samples, self.contacts, self.targets, self.content_labels = self.make_dataset(opt)

    def make_dataset(self, opt):
        X1, X2, F1, F2, Y, C1, C2 = [], [], [], [], [], [], []
        for dom in range(opt.num_domains): #style domain
            dom_idx = [si for si in range(len(self.classes))
                       if self.classes[si][1] == styles.index(opt.domains[dom])]  # index list that belongs to the domain
            dom_idx2 = random.sample(dom_idx, len(dom_idx))
            dom_clips = [self.clips[cli] for cli in dom_idx]  # clips list (motion data) that belongs to the domain
            dom_feet = [self.feet[fti] for fti in dom_idx]
            dom_clips2 = [self.clips[cli] for cli in dom_idx2]
            dom_feet2 = [self.feet[fti] for fti in dom_idx2]
            dom_contents = [self.classes[ci][0] for ci in dom_idx]
            dom_contents2 = [self.classes[ci][0] for ci in dom_idx2]
            X1 += dom_clips
            X2 += dom_clips2
            F1 += dom_feet
            F2 += dom_feet2
            Y += [dom] * len(dom_clips)
            C1 += dom_contents
            C2 += dom_contents2
        return list(zip(X1, X2)), list(zip(F1, F2)), Y, list(zip(C1, C2))

    def __getitem__(self, index):
        x, x2 = self.samples[index]
        f, f2 = self.contacts[index]
        if self.opt.motion_length > 20:
            roundlen = self.opt.motion_length
        else:
            roundlen = x.shape[1] // 16 * 16
        idx = random.randint(0, x.shape[1] - roundlen)
        # print(f.shape, f2.shape)
        x = x[:, idx:idx+roundlen]
        f = f[idx:idx+roundlen]
        x2 = x2[:, idx:idx+roundlen]
        f2 = f2[idx:idx+roundlen]
        # print(f.shape, f2.shape)
        x = normalize(x, self.clips_mean, self.clips_std)
        x2 = normalize(x2, self.clips_mean, self.clips_std)
        x_data = {'posrot': x[:9], 'traj': x[-4:], 'feet': f}
        x2_data = {'posrot': x2[:9], 'traj': x2[-4:], 'feet': f}
        y = self.targets[index]
        c, c2 = self.content_labels[index]
        return {'x': x_data, 'x2': x2_data, 'y': y, 'c': c, 'c2': c2}

    def __len__(self):
        return len(self.targets)


class InputFetcher:
    def __init__(self, opt, loader, loader_ref=None):
        self.loader = loader
        self.loader_ref = loader_ref
        self.latent_dim = opt.latent_dim
        self.device = torch.device("cuda:{}".format(opt.gpu_ids[0]) if torch.cuda.is_available() else "cpu")

    def fetch_src(self):
        try:
            src = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            src = next(self.iter)
        return src

    def fetch_refs(self):
        try:
            ref = next(self.iter_ref)
        except (AttributeError, StopIteration):
            self.iter_ref = iter(self.loader_ref)
            ref = next(self.iter_ref)
        return ref

    def __next__(self):
        inputs = {}
        src = self.fetch_src()
        inputs_src = {'x_real': src['x'], 'y_org': src['y'], 'c_real': src['c']}
        inputs.update(inputs_src)

        if self.loader_ref is not None:
            ref = self.fetch_refs()
            z = torch.randn(src['y'].size(0), self.latent_dim)   # random Gaussian noise for x_ref
            z2 = torch.randn(src['y'].size(0), self.latent_dim)  # random Gaussian noise for x_ref2
            inputs_ref = {'x_ref': ref['x'], 'x_ref2': ref['x2'],
                          'c_ref': ref['c'], 'c_ref2': ref['c2'],
                          'y_trg': ref['y'],
                          'z_trg': z, 'z_trg2': z2}
            inputs.update(inputs_ref)
        return to(inputs, self.device)

class TestInputFetcher:
    def __init__(self, opt, loader, loader_ref=None):
        self.loader = loader
        self.loader_ref = loader_ref
        self.latent_dim = opt.latent_dim
        self.device = torch.device("cuda:{}".format(opt.gpu_ids[0]) if torch.cuda.is_available() else "cpu")

    def fetch_src(self):
        try:
            src = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            src = next(self.iter)
        return src

    def fetch_refs(self):
        try:
            ref = next(self.iter_ref)
        except (AttributeError, StopIteration):
            self.iter_ref = iter(self.loader_ref)
            ref = next(self.iter_ref)
        return ref

    def __next__(self):
        inputs = {}
        src = self.fetch_src()
        inputs_src = {'x_real': src['x'], 'y_org': src['y'], 'c_real': src['c']}
        inputs.update(inputs_src)

        if self.loader_ref is not None:
            ref = self.fetch_refs()
            while ref['y'] == inputs_src['y_org']: #ensure a different target style
                ref = self.fetch_refs()
            z = torch.randn(src['y'].size(0), self.latent_dim)   # random Gaussian noise for x_ref
            inputs_ref = {'x_ref': ref['x'],
                          'c_ref': ref['c'],
                          'y_trg': ref['y'],
                          'z_trg': z}
            inputs.update(inputs_ref)
        return to(inputs, self.device)

    def repeat(self, data, num_repeats):
        B = data['z_trg'].shape[0]
        inputs = {}
        for k, v in data.items():
            if k == 'z_trg':
                z = torch.randn(B*num_repeats, self.latent_dim).to(self.device)
                inputs['z_trg'] = z
            elif isinstance(v, dict):
                inputs[k] = {}
                for k1, v1 in v.items():
                    inputs[k][k1] = v1.repeat_interleave(num_repeats, 0)
            else:
                # inputs[k] = v.repeat(*([num_repeats]+list(v.shape[1:])))
                inputs[k] = v.repeat_interleave(num_repeats, 0)
        return inputs
                

# class TestInputFetcher:
#     def __init__(self, opt):
#         self.latent_dim = opt.latent_dim
#         self.preprocess = np.load(opt.preproot)
#         self.device = torch.device("cuda:{}".format(opt.gpu_ids[0]) if torch.cuda.is_available() else "cpu")

#     def get_data(self, filename, sty, con, start=0, end=None, downsample=1, type='src'):
#         data, feet = generate_data(filename, downsample=downsample)  # real data
#         if end is not None:
#             data = data[:, start:end, :]
#             feet = feet[start:end, :]

#         x = normalize(data, self.preprocess['Xmean'], self.preprocess['Xstd'])
#         x = torch.tensor(x, dtype=torch.float)
#         f = torch.tensor(feet, dtype=torch.float)
#         y = torch.tensor(sty, dtype=torch.long)
#         c = con
#         x_data = {'posrot': x[:9], 'traj': x[-4:], 'feet': f}

#         if type == 'src':
#             src = {'x': x_data, 'y': y, 'c': c}
#             input = {'x_real': src['x'], 'y_org': src['y'], 'c_real': src['c']}
#         elif type == 'ref':
#             ref = {'x': x_data, 'y': y, 'c': c}
#             input = {'x_ref': ref['x'], 'y_trg': ref['y'], 'c_ref': ref['c']}

#         return to(input, self.device, expand_dim=True)

#     def get_latent(self):
#         inputs = {}
#         z_trg = torch.randn(self.latent_dim)
#         inputs_latent = {'z_trg': z_trg}
#         inputs.update(inputs_latent)
#         return to(inputs, self.device, expand_dim=True)


def to(inputs, device, expand_dim=False):
    for name, ele in inputs.items():
        if isinstance(ele, dict):
            for k, v in ele.items():
                if expand_dim:
                    v = torch.unsqueeze(torch.tensor(v), dim=0)
                ele[k] = v.to(device, dtype=torch.float)
        else:
            if expand_dim:
                ele = torch.unsqueeze(torch.tensor(ele), dim=0)
            if name.startswith('y_') or name.startswith('c_'):
                inputs[name] = ele.to(device, dtype=torch.long)
            else:
                inputs[name] = ele.to(device, dtype=torch.float)
    return inputs
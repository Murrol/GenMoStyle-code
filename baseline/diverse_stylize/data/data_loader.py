import sys
import numpy as np
import random
import torch
from torch.utils import data

sys.path.append('../')
from diverse_utils.helper import normalize
from preprocess.export_dataset import generate_data


f = open('contents.txt', 'r')
contents = [line.strip() for line in f.readlines()]

f = open('styles.txt', 'r')
styles = [line.strip() for line in f.readlines()]


class SourceDataset(data.Dataset):
    def __init__(self, opt, type='train'):
        if type == 'train':
            dataroot = opt.train_dataroot
        elif type == 'test':
            dataroot = opt.test_dataroot
        
        self.domains = opt.domains
        self.clips = np.load(dataroot)['clips']
        self.feet = np.load(dataroot, allow_pickle=True)['feet']
        self.classes = np.load(dataroot)['classes']

        self.preprocess = np.load(opt.preproot)
        self.samples, self.contacts, self.targets, self.labels = self.make_dataset(opt)

    def make_dataset(self, opt):
        X, F, Y, C = [], [], [], []
        for dom in range(opt.num_domains):
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
        x = normalize(x, self.preprocess['Xmean'], self.preprocess['Xstd'])
        data = {'posrot': x[:7], 'traj': x[-4:], 'feet': f}
        y = self.targets[index]
        c = self.labels[index]
        return {'x': data, 'y': y, 'c': c}

    def __len__(self):
        return len(self.targets)


class ReferenceDataset(data.Dataset):
    def __init__(self, opt, type='train'):
        if type == 'train':
            dataroot = opt.train_dataroot
        elif type == 'test':
            dataroot = opt.test_dataroot

        self.domains = opt.domains
        self.clips = np.load(dataroot)['clips']
        self.classes = np.load(dataroot)['classes']
        self.preprocess = np.load(opt.preproot)
        self.samples, self.contacts, self.targets, self.labels = self.make_dataset(opt)

    def make_dataset(self, opt):
        X1, X2, F1, F2, Y, C1, C2 = [], [], [], [], [], [], []
        for dom in range(opt.num_domains):
            dom_idx = [si for si in range(len(self.classes))
                       if self.classes[si][1] == styles.index(opt.domains[dom])]  # index list that belongs to the domain
            dom_idx2 = random.sample(dom_idx, len(dom_idx))
            dom_clips = [self.clips[cli] for cli in dom_idx]  # clips list (motion data) that belongs to the domain
            dom_feet = [self.clips[fti] for fti in dom_idx]
            dom_clips2 = [self.clips[cli] for cli in dom_idx2]
            dom_feet2 = [self.clips[fti] for fti in dom_idx2]
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
        x = normalize(x, self.preprocess['Xmean'], self.preprocess['Xstd'])
        x2 = normalize(x2, self.preprocess['Xmean'], self.preprocess['Xstd'])
        x_data = {'posrot': x[:7], 'traj': x[-4:], 'feet': f}
        x2_data = {'posrot': x2[:7], 'traj': x2[-4:], 'feet': f}
        y = self.targets[index]
        c, c2 = self.labels[index]
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
    def __init__(self, opt):
        self.latent_dim = opt.latent_dim
        self.preprocess = np.load(opt.preproot)
        self.device = torch.device("cuda:{}".format(opt.gpu_ids[0]) if torch.cuda.is_available() else "cpu")

    def get_data(self, filename, sty, con, start=0, end=None, downsample=1, type='src'):
        data, feet = generate_data(filename, downsample=downsample)  # real data
        if end is not None:
            data = data[:, start:end, :]
            feet = feet[start:end, :]

        x = normalize(data, self.preprocess['Xmean'], self.preprocess['Xstd'])
        x = torch.tensor(x, dtype=torch.float)
        f = torch.tensor(feet, dtype=torch.float)
        y = torch.tensor(sty, dtype=torch.long)
        c = con
        x_data = {'posrot': x[:7], 'traj': x[-4:], 'feet': f}

        if type == 'src':
            src = {'x': x_data, 'y': y, 'c': c}
            input = {'x_real': src['x'], 'y_org': src['y'], 'c_real': src['c']}
        elif type == 'ref':
            ref = {'x': x_data, 'y': y, 'c': c}
            input = {'x_ref': ref['x'], 'y_trg': ref['y'], 'c_ref': ref['c']}

        return to(input, self.device, expand_dim=True)

    def get_latent(self):
        inputs = {}
        z_trg = torch.randn(self.latent_dim)
        inputs_latent = {'z_trg': z_trg}
        inputs.update(inputs_latent)
        return to(inputs, self.device, expand_dim=True)


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
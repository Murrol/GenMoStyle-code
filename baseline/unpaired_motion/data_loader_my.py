import collections
import os
import sys
import random
import torch
import numpy as np
import argparse
BASEPATH = os.path.dirname(__file__)
from os.path import join as pjoin
sys.path.insert(0, BASEPATH)
sys.path.insert(0, pjoin(BASEPATH, '..'))

from torch.utils.data import Dataset, DataLoader
from animation_data import AnimationData
from load_skeleton import Skel
from config import Config
from py_utils import print_composite

from scripts.motion_process_bvh import *

def normalize_motion(motion, mean_pose, std_pose):
    """
    inputs:
    motion: (V, C, T) or (C, T)
    mean_pose: (C, 1)
    std_pose: (C, 1)
    """
    return (motion - mean_pose) / std_pose


class NormData:
    def __init__(self, name, pre_computed, raw, config, data_dir, keep_raw=False): # data_dir stores the .npz
        """
        raw:
        - nrot: N * [J * 4 + 4, T]
        - rfree: N * [J * 3 + 4, T]
        - proj: N * [V, J * 2, T]
        """

        self.norm_path = os.path.join(data_dir, name + ".npz")
        if os.path.exists(self.norm_path):
            print("loading norm from file: %s" % self.norm_path)
            norm = np.load(self.norm_path, allow_pickle=True)
            self.mean, self.std = norm['mean'], norm['std']
        else:
            if pre_computed:
                assert 0, f'Error! {self.norm_path} not found!'
            # a list of [V, J * 2, T] / [J * 3/4 + 4, T]
            # --> [V, J * 2, sumT] / [J * 3/4 + 4, sumT]
            data = np.concatenate(raw, axis=-1)
            print("data shape 1/3:", data.shape)
            # [V, sumT, J * 2] / [sumT, J * 3/4 + 4]
            data = data.swapaxes(-1, -2)
            print("data shape 2/3:", data.shape)
            # [V * sumT, J * 2] / [sumT, J * 3/4 + 4]
            data = data.reshape((-1, data.shape[-1]))
            print("data shape 3/3:", data.shape)

            self.mean = np.mean(data, axis=0)
            self.std = np.std(data, axis=0)
            self.std[np.where(self.std == 0)] = 1e-9
            np.savez(self.norm_path, mean=self.mean, std=self.std)
            print("mean and std saved at {}".format(self.norm_path))

        self.mean = torch.tensor(self.mean, dtype=torch.float, device=config.device).unsqueeze(-1)
        self.std = torch.tensor(self.std, dtype=torch.float, device=config.device).unsqueeze(-1) # [C, 1]
        if keep_raw:
            self.raw = raw
            self.norm = {}
            for i in range(len(self.raw)):
                self.raw[i] = torch.tensor(self.raw[i], dtype=torch.float, device=config.device)

    def get_raw(self, index):
        return self.raw[index]

    def get_norm(self, index):
        if index not in self.norm:
            self.norm[index] = normalize_motion(self.raw[index], self.mean, self.std)
        return self.norm[index]

    def normalize(self, raw):
        return normalize_motion(raw, self.mean, self.std)


class MotionNorm(Dataset):
    def __init__(self, config, subset_name, data_path=None, extra_data_dir=None):
        super(MotionNorm, self).__init__()

        np.random.seed(2020)
        self.skel = Skel()  # TD: add config
        self.skeleton = Skeleton(self.skel.offset, self.skel.topology, "cpu")
        self.config = config
        joints_num = 21
        # motion_length = 300
        if data_path is None:
            if "train" in subset_name:
                data_path = pjoin(config.data_dir, "train_data.npy")
            elif "test" in subset_name:
                data_path = pjoin(config.data_dir, "test_data.npy")
            # data_path = config.data_path
        # dataset = np.load(data_path, allow_pickle=True)[subset_name].item()
        # motions, labels, metas = dataset["motion"], dataset["style"], dataset["meta"]

        data = np.load(data_path, allow_pickle=True).item()
        new_data = {}
        name_list = []
        labels, actions = [], []

        style_dict = collections.defaultdict(list)
        sequence_dict = collections.defaultdict(list)

        for key, value in data.items():
            if len(value) < config.motion_length:
            # if len(value) < motion_length:
                continue
            new_data[key] = value #(300, 260) 
            name_list.append(key)
            style_id = key.split("#")[-1]
            sequence_id = key.split("#")[0]
            if sequence_id.startswith("m_"):
                sequence_id = sequence_id[2:]
            style_dict[style_id].append(key)
            sequence_dict[sequence_id].append(key)
            # labels.append(eval(style_id))
            if "cmu" in config.data_dir:
                labels.append(0)
            else:
                labels.append(eval(key.split("#")[-1]))
            if "xia" in config.data_dir:
                actions.append(eval(key.split("#")[-2]))
            else:
                actions.append(0)

        # print(config.data_dir, labels)
        self.label_i = labels
        self.action_i = actions
        self.data_dict_raw = new_data
        self.name_list = name_list
        self.style_dict = style_dict
        self.sequence_dict = sequence_dict

        self.len = len(self.name_list)
        self.motion_i, self.foot_i = [], []
        content, style3d, style2d = [], [], []

        self.labels = []
        self.data_dict = {}
        self.diff_labels_dict = {}
        for i, name in enumerate(name_list):
            label = eval(name.split("#")[-1])
            value = self.data_dict_raw[name]
            _, lq, rp = recover_bvh_from_rot(torch.from_numpy(value).float(), joints_num, self.skeleton)
            anim = AnimationData.from_rotations_and_root_positions(rotations=lq.numpy(), root_positions=rp.numpy(), skel=self.skel, frametime=1/30)
            if label not in self.labels:
                self.labels.append(label)
                self.data_dict[label] = []
            self.data_dict[label].append(i)
            self.motion_i.append(anim)
            self.foot_i.append(anim.get_foot_contact(transpose=True))  # [4, T]
            content.append(anim.get_content_input())
            style3d.append(anim.get_style3d_input())
            view_angles, scales = [], []
            for v in range(10):
                view_angles.append(self.random_view_angle())
                scales.append(self.random_scale())
            style2d.append(anim.get_projections(view_angles, scales))

        # calc diff labels
        for x in self.labels:
            self.diff_labels_dict[x] = [y for y in self.labels if y != x]

        if extra_data_dir is None:
            extra_data_dir = config.extra_data_dir

        norm_cfg = config.dataset_norm_config
        norm_data = []
        for key, raw in zip(["content", "style3d", "style2d"], [content, style3d, style2d]):
            prefix = norm_cfg[subset_name][key]
            pre_computed = prefix is not None
            if prefix is None:
                prefix = subset_name
            norm_data.append(NormData(prefix + "_" + key, pre_computed, raw,
                                      config, extra_data_dir, keep_raw=(key != "style2d")))
        self.content, self.style3d, self.style2d = norm_data
        self.device = config.device
        self.rand = random.SystemRandom()

    @staticmethod
    def random_view_angle():
        return (0, -np.pi / 2 + float(np.random.rand(1)) * np.pi, 0)

    @staticmethod
    def random_scale():
        return float(np.random.rand(1)) * 0.4 + 0.8

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        label = self.label_i[index]
        action = self.action_i[index]
        
        if len(self.diff_labels_dict[label]) == 0:
            l_diff = label
        else:
            l_diff = self.rand.choice(self.diff_labels_dict[label])
        index_same = self.rand.choice(self.data_dict[label])
        index_diff = self.rand.choice(self.data_dict[l_diff])
        # assert self.foot_i[index].shape[-1] >= self.config.motion_length
        if self.config.motion_length > 20:
            roundlen = self.config.motion_length
            idx_crop = random.randint(0, self.foot_i[index].shape[-1] - roundlen)
        else:
            roundlen = self.foot_i[index].shape[-1] // 16 * 16
            idx_crop = random.randint(0, self.foot_i[index].shape[-1] - roundlen)
        

        data = {"label": label,
                "action": action,
                "label_diff": l_diff,
                "foot_contact": self.foot_i[index][..., idx_crop:idx_crop+roundlen],  # for footskate fixing
                "content": self.content.get_norm(index)[..., idx_crop:idx_crop+roundlen],
                "style3d": self.style3d.get_norm(index)[..., idx_crop:idx_crop+roundlen],
                "contentraw": self.content.get_raw(index)[..., idx_crop:idx_crop+roundlen],  # for visualization
                "style3draw": self.style3d.get_raw(index)[..., idx_crop:idx_crop+roundlen],  # positions are used as the recon target
                "same_style3d": self.style3d.get_norm(index_same)[..., idx_crop:idx_crop+roundlen],
                "diff_style3d": self.style3d.get_norm(index_diff)[..., idx_crop:idx_crop+roundlen],
                "diff_style3d_nrot": self.content.get_raw(index_diff)[..., idx_crop:idx_crop+roundlen],
                }

        for idx, key in zip([index, index_same, index_diff], ["style2d", "same_style2d", "diff_style2d"]):
            raw = self.motion_i[idx].get_projections((self.random_view_angle(),), (self.random_scale(),))[0]
            raw = torch.tensor(raw, dtype=torch.float, device=self.device)
            if key == "style2d":
                data[key + "raw"] = raw[..., idx_crop:idx_crop+roundlen]
            data[key] = self.style2d.normalize(raw[..., idx_crop:idx_crop+roundlen])

        return data


def get_dataloader(config, subset_name, shuffle=None,
                   data_path=None, extra_data_dir=None, drop_last=False):
    dataset = MotionNorm(config, subset_name, data_path=data_path, extra_data_dir=extra_data_dir)
    batch_size = config.batch_size if subset_name == 'train' else 1
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=(subset_name == 'train') if shuffle is None else shuffle,
                      num_workers=0, drop_last=drop_last)


def single_to_batch(data):
    for key, value in data.items():
        if key == "meta":
            data[key] = {sub_key: [sub_value] for sub_key, sub_value in value.items()}
        else:
            data[key] = value.unsqueeze(0)
    return data


def process_single_bvh(filename, config, norm_data_dir=None, downsample=4, skel=None, to_batch=False):
    def to_tensor(x):
        return torch.tensor(x).float().to(config.device)

    anim = AnimationData.from_BVH(filename, downsample=downsample, skel=skel, trim_scale=4)
    foot_contact = anim.get_foot_contact(transpose=True)  # [4, T]
    content = to_tensor(anim.get_content_input())
    style3d = to_tensor(anim.get_style3d_input())

    data = {"meta": {"style": "test", "content": filename.split('/')[-1]},
            "foot_contact": to_tensor(foot_contact),
            "contentraw": content,
            "style3draw": style3d
            }

    if norm_data_dir is None:
        norm_data_dir = config.extra_data_dir
    for key, raw in zip(["content", "style3d"], [content, style3d]):
        norm_path = os.path.join(norm_data_dir, f'train_{key}.npz')
        norm = np.load(norm_path, allow_pickle=True)
        data[key] = normalize_motion(raw,
                                     to_tensor(norm['mean']).unsqueeze(-1),
                                     to_tensor(norm['std']).unsqueeze(-1))

    if to_batch:
        data = single_to_batch(data)

    return data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--config', type=str, default='config')
    return parser.parse_args()


def test_dataset(args):
    config = Config()
    config.initialize(args)
    data = process_single_bvh('data_proc/styletransfer/proud_03_001.bvh', config, to_batch=True)
    print_composite(data)

    """
    train_dataset = MotionNorm(config, "train")

    print_composite(train_dataset[0])
    data_loader = DataLoader(train_dataset, batch_size=2, shuffle=False)
    for batch in data_loader:
        print_composite(batch)
        break
    """


if __name__ == '__main__':
    args = parse_args()
    test_dataset(args)

import collections

import torch
from torch.utils import data
import numpy as np
import os
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm

class MotionDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file_path, fix_bias=False):
        self.opt = opt
        # joint_num = opt.joint_n

        data = np.load(split_file_path, allow_pickle=True).item()
        new_data = {}
        name_list = []
        style_dict = collections.defaultdict(list)
        sequence_dict = collections.defaultdict(list)

        for key, value in data.items():
            if len(value) < opt.motion_length:
                continue
            new_data[key] = value
            name_list.append(key)
            style_id = key.split("#")[-1]
            sequence_id = key.split("#")[0]
            if sequence_id.startswith("m_"):
                sequence_id = sequence_id[2:]
            style_dict[style_id].append(key)
            sequence_dict[sequence_id].append(key)
        if opt.is_train and (not fix_bias):
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4: 4 + opt.joint_num * 3] = std[4: 4 + opt.joint_num * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + opt.joint_num * 3: 4 + opt.joint_num * 9] = std[4 + opt.joint_num * 3:
                                                                      4 + opt.joint_num * 9] / 1.0
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + opt.joint_num * 9:
                4 + opt.joint_num * 12] = std[4 + opt.joint_num * 9:
                                               4 + opt.joint_num * 12] / 1.0
            # foot contact (B, seq_len, 4)
            std[4 + opt.joint_num * 12:] = std[4 + opt.joint_num * 12:] / opt.feat_bias

            assert 4 + opt.joint_num * 12 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
            np.save(pjoin(opt.meta_dir, 'std.npy'), std)
        self.mean = mean
        self.std = std
        self.data_dict = new_data
        self.name_list = name_list
        self.style_dict = style_dict
        self.sequence_dict = sequence_dict

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.name_list)

    # root_rot_velocity (seq_len, 1)
    # root_linear_velocity (seq_len, 2)
    # root_y (seq_len, 1)
    # ric_data (seq_len, joint_num * 3)
    # rot_data (seq_len, joint_num * 6)
    # local_velocity (seq_len, joint_num*3)
    # foot contact (seq_len, 4)
    def __skeletonize(self, motion):
        joint_num = self.opt.joint_num
        root_data = motion[:, :4]
        positions = motion[:, 4:4 + joint_num * 3]
        rotations = motion[:, 4 + joint_num * 3 : 4 + joint_num * 9]
        velocities = motion[:, 4 + joint_num * 9 : 4 + joint_num * 12]
        foot_contact = motion[:, 4 + joint_num * 12:]
        positions = positions.reshape(-1, joint_num, 3)
        rotations = rotations.reshape(-1, joint_num, 6)
        velocities = velocities.reshape(-1, joint_num, 3)
        root_data = root_data[:, np.newaxis].repeat(joint_num, axis=1)
        foot_contact = foot_contact[:, np.newaxis].repeat(joint_num, axis=1)
        data = np.concatenate([positions, rotations, velocities, root_data, foot_contact], axis=-1)
        data = data.reshape(len(motion), -1)
        return data

    def deskeletonize(self, motion):
        joint_num = self.opt.joint_num
        shape = motion.shape
        motion = motion.reshape(shape[:-1] + (joint_num, -1))
        #     print(motion.shape)
        positions = motion[..., :3].reshape(shape[:-1] + (-1,))
        rotations = motion[..., 3:9].reshape(shape[:-1] + (-1,))
        velocities = motion[..., 9:12].reshape(shape[:-1] + (-1,))
        root_data = motion[..., 0, 12:16]
        foot_contact = motion[..., 0, 16:20]
        #     print(positions.shape)
        data = np.concatenate([root_data, positions, rotations, velocities, foot_contact], axis=-1)
        return data

    def __getitem__(self, item):
        name = self.name_list[item]
        # style_id = self.style_dict[name]
        style_id = name.split("#")[-1]
        sequence_id = name.split("#")[0]
        sequence_id = sequence_id[2:] if sequence_id.startswith("m_") else sequence_id

        motion = self.data_dict[name]
        # Motion from the same sequence
        motion1 = self.data_dict[random.choice(self.sequence_dict[sequence_id])]
        # Motion from the same style
        motion2 = self.data_dict[random.choice(self.style_dict[style_id])]
        # Motion from different style
        another_style = random.choice([style for style in self.style_dict.keys() if style != style_id])
        # name2 = random.choice(self.style_dict[style_id])
        motion3 = self.data_dict[random.choice(self.style_dict[another_style])]

        # print("Before", motion[0])
        # print(self.mean.mean())
        # print(self.std.mean())

        # Mask out root velecity information
        # motion[:, 1:3] *= 0
        """Z Normalization"""
        motion = (motion - self.mean) / self.std
        motion1 = (motion1 - self.mean) / self.std
        motion2 = (motion2 - self.mean) / self.std
        motion3 = (motion3 - self.mean) / self.std
        # print("After", motion[0])

        # m_length = data.shape[0]
        assert len(motion) >= self.opt.motion_length
        idx = random.randint(0, len(motion) - self.opt.motion_length)
        # idx2 = random.randint(0, len(motion) - self.opt.motion_length)

        data = motion[idx:idx+self.opt.motion_length]
        # Motion from the same sequence
        data1 = motion1[idx:idx+self.opt.motion_length]
        # Motion from the same style
        data2 = motion2[idx:idx+self.opt.motion_length]
        # Motion from different style
        data3 = motion3[idx:idx + self.opt.motion_length]
        # # Motion from different style
        # data4 = motion2[idx2:idx2+self.opt.motion_length]


        style_id1 = int(style_id)
        style_id3 = int(another_style)
        # BFA dataset does not contain action information
        action_id = 0
        if self.opt.dataset_name == "xia":
            action_id = int(name.split("#")[-2])

        style_one_hot1 = np.zeros(self.opt.num_of_style)
        style_one_hot1[style_id1] = 1

        style_one_hot3 = np.zeros(self.opt.num_of_style)
        style_one_hot3[style_id3] = 1

        action_one_hot = np.zeros(self.opt.num_of_action)
        action_one_hot[action_id] = 1

        return data, data1, data2, data3, action_one_hot, style_one_hot1, style_id1, style_one_hot3, style_id3


class MotionRegressorDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file_path):
        self.opt = opt
        # joint_num = opt.joint_n

        data = np.load(split_file_path, allow_pickle=True).item()
        new_data = {}
        name_list = []
        for key, value in data.items():
            if len(value) < opt.motion_length:
                continue
            new_data[key] = value
            name_list.append(key)
        self.mean = mean[..., :-4]
        self.std = std[..., :-4]
        self.data_dict = new_data
        self.name_list = name_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.name_list)

    # root_rot_velocity (seq_len, 1)
    # root_linear_velocity (seq_len, 2)
    # root_y (seq_len, 1)
    # ric_data (seq_len, joint_num * 3)
    # rot_data (seq_len, joint_num * 6)
    # local_velocity (seq_len, joint_num*3)
    # foot contact (seq_len, 4)
    def __skeletonize(self, motion):
        joint_num = self.opt.joint_num
        root_data = motion[:, :4]
        positions = motion[:, 4:4 + joint_num * 3]
        rotations = motion[:, 4 + joint_num * 3 : 4 + joint_num * 9]
        velocities = motion[:, 4 + joint_num * 9 : 4 + joint_num * 12]
        foot_contact = motion[:, 4 + joint_num * 12:]
        positions = positions.reshape(-1, joint_num, 3)
        rotations = rotations.reshape(-1, joint_num, 6)
        velocities = velocities.reshape(-1, joint_num, 3)
        root_data = root_data[:, np.newaxis].repeat(joint_num, axis=1)
        foot_contact = foot_contact[:, np.newaxis].repeat(joint_num, axis=1)
        data = np.concatenate([positions, rotations, velocities, root_data, foot_contact], axis=-1)
        data = data.reshape(len(motion), -1)
        return data

    def deskeletonize(self, motion):
        joint_num = self.opt.joint_num
        shape = motion.shape
        motion = motion.reshape(shape[:-1] + (joint_num, -1))
        #     print(motion.shape)
        positions = motion[..., :3].reshape(shape[:-1] + (-1,))
        rotations = motion[..., 3:9].reshape(shape[:-1] + (-1,))
        velocities = motion[..., 9:12].reshape(shape[:-1] + (-1,))
        root_data = motion[..., 0, 12:16]
        foot_contact = motion[..., 0, 16:20]
        #     print(positions.shape)
        data = np.concatenate([root_data, positions, rotations, velocities, foot_contact], axis=-1)
        return data

    def __getitem__(self, item):
        name = self.name_list[item]
        motion = self.data_dict[name]
        motion = motion[..., :-4]
        # print("Before", motion[0])
        # print(self.mean.mean())
        # print(self.std.mean())

        # Mask out root velecity information
        # motion[:, 1:3] *= 0
        """Z Normalization"""
        motion = (motion - self.mean) / self.std
        # print("After", motion[0])
        return motion


class MotionEvalDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file_path):
        self.opt = opt
        # joint_num = opt.joint_n

        data = np.load(split_file_path, allow_pickle=True).item()
        new_data = {}
        name_list = []
        self.style_dict = {}
        for key, value in data.items():
            if len(value) < opt.motion_length:
                continue
            new_data[key] = value
            name_list.append(key)
            style_label = int(key.split("#")[-1])
            if style_label not in self.style_dict:
                self.style_dict[style_label] = [len(name_list)-1]
            else:
                self.style_dict[style_label].append(len(name_list)-1)
        self.mean = mean
        self.std = std
        self.data_dict = new_data
        self.name_list = name_list
        self.fix_content_id = False
        self.fix_style = False
        # self.content_id = 62

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.name_list)

    # root_rot_velocity (seq_len, 1)
    # root_linear_velocity (seq_len, 2)
    # root_y (seq_len, 1)
    # ric_data (seq_len, joint_num * 3)
    # rot_data (seq_len, joint_num * 6)
    # local_velocity (seq_len, joint_num*3)
    # foot contact (seq_len, 4)
    def __skeletonize(self, motion):
        joint_num = self.opt.joint_num
        root_data = motion[:, :4]
        positions = motion[:, 4:4 + joint_num * 3]
        rotations = motion[:, 4 + joint_num * 3 : 4 + joint_num * 9]
        velocities = motion[:, 4 + joint_num * 9 : 4 + joint_num * 12]
        foot_contact = motion[:, 4 + joint_num * 12:]
        positions = positions.reshape(-1, joint_num, 3)
        rotations = rotations.reshape(-1, joint_num, 6)
        velocities = velocities.reshape(-1, joint_num, 3)
        root_data = root_data[:, np.newaxis].repeat(joint_num, axis=1)
        foot_contact = foot_contact[:, np.newaxis].repeat(joint_num, axis=1)
        data = np.concatenate([positions, rotations, velocities, root_data, foot_contact], axis=-1)
        data = data.reshape(len(motion), -1)
        return data

    def deskeletonize(self, motion):
        joint_num = self.opt.joint_num
        shape = motion.shape
        motion = motion.reshape(shape[:-1] + (joint_num, -1))
        #     print(motion.shape)
        positions = motion[..., :3].reshape(shape[:-1] + (-1,))
        rotations = motion[..., 3:9].reshape(shape[:-1] + (-1,))
        velocities = motion[..., 9:12].reshape(shape[:-1] + (-1,))
        root_data = motion[..., 0, 12:16]
        foot_contact = motion[..., 0, 16:20]
        #     print(positions.shape)
        data = np.concatenate([root_data, positions, rotations, velocities, foot_contact], axis=-1)
        return data

    def set_style(self, style_c, style_r):
        self.style_c = style_c
        self.style_r = style_r
        self.fix_style = True

    def set_content_id(self, content_id):
        self.content_id = content_id
        self.fix_content_id = True

    def __getitem__(self, item):
        if self.fix_content_id:
            name1 = self.name_list[self.content_id]
            M1 = self.data_dict[name1][:-4]
            # M1 = M1[len(M1)-120:]
            name2 = self.name_list[item]
            M2 = self.data_dict[name2]
            idx2 = random.randint(0, len(M2) - self.opt.motion_length)
            M2 = M2[idx2:idx2+self.opt.motion_length]
        else:
            name1 = self.name_list[random.choice(self.style_dict[self.style_c])]
            name2 = self.name_list[random.choice(self.style_dict[self.style_r])]
            M1 = self.data_dict[name1]
            M2 = self.data_dict[name2]
            idx1 = random.randint(0, len(M1) - self.opt.motion_length)
            idx2 = random.randint(0, len(M2) - self.opt.motion_length)
            M1 = M1[idx1:idx1+self.opt.motion_length]
            M2 = M2[idx2:idx2+self.opt.motion_length]

        """Z Normalization"""
        M1 = (M1 - self.mean) / self.std
        M2 = (M2 - self.mean) / self.std
        # print("After", motion[0])

        style_id1 = int(name1.split("#")[-1])
        style_id2 = int(name2.split("#")[-1])

        # BFA dataset does not contain action information
        action_id1, action_id2 = 0, 0
        if self.opt.dataset_name == "xia":
            action_id1 = int(name1.split("#")[-2])
            action_id2 = int(name2.split("#")[-2])

        style_one_hot1 = np.zeros(self.opt.num_of_style)
        style_one_hot1[style_id1] = 1
        style_one_hot2 = np.zeros(self.opt.num_of_style)
        style_one_hot2[style_id2] = 1

        action_one_hot1 = np.zeros(self.opt.num_of_action)
        action_one_hot1[action_id1] = 1
        action_one_hot2 = np.zeros(self.opt.num_of_action)
        action_one_hot2[action_id2] = 1

        return M1, M2, action_one_hot1, action_one_hot2, style_one_hot1, style_one_hot2, style_id1, style_id2


class MotionXiaDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file_path, fix_bias=False):
        self.opt = opt
        # joint_num = opt.joint_n

        data = np.load(split_file_path, allow_pickle=True).item()
        new_data = {}
        name_list = []

        for key, value in data.items():
            # if len(value) < opt.motion_length:
            #     continue
            if len(value) <= 16:
                continue
            new_data[key] = value
            name_list.append(key)
            # style_id = key.split("#")[-1]
            # sequence_id = key.split("#")[0]
            # if sequence_id.startswith("m_"):
            #     sequence_id = sequence_id[2:]
            # style_dict[style_id].append(key)
            # sequence_dict[sequence_id].append(key)
        if opt.is_train and (not fix_bias):
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4: 4 + opt.joint_num * 3] = std[4: 4 + opt.joint_num * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + opt.joint_num * 3: 4 + opt.joint_num * 9] = std[4 + opt.joint_num * 3:
                                                                      4 + opt.joint_num * 9] / 1.0
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + opt.joint_num * 9:
                4 + opt.joint_num * 12] = std[4 + opt.joint_num * 9:
                                               4 + opt.joint_num * 12] / 1.0
            # foot contact (B, seq_len, 4)
            std[4 + opt.joint_num * 12:] = std[4 + opt.joint_num * 12:] / opt.feat_bias

            assert 4 + opt.joint_num * 12 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
            np.save(pjoin(opt.meta_dir, 'std.npy'), std)
        self.mean = mean
        self.std = std
        self.data_dict = new_data
        self.name_list = name_list
        # self.style_dict = style_dict
        # self.sequence_dict = sequence_dict

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, item):
        name = self.name_list[item]
        # style_id = self.style_dict[name]
        # style_id = name.split("#")[-1]
        # sequence_id = name.split("#")[0]
        # sequence_id = sequence_id[2:] if sequence_id.startswith("m_") else sequence_id

        motion = self.data_dict[name]
        # Motion from the same sequence
        # motion1 = self.data_dict[random.choice(self.sequence_dict[sequence_id])]
        # Motion from the same style
        # motion2 = self.data_dict[random.choice(self.style_dict[style_id])]
        # Motion from different style
        # another_style = random.choice([style for style in self.style_dict.keys() if style != style_id])
        # name2 = random.choice(self.style_dict[style_id])
        # motion3 = self.data_dict[random.choice(self.style_dict[another_style])]

        # print("Before", motion[0])
        # print(self.mean.mean())
        # print(self.std.mean())

        # Mask out root velecity information
        # motion[:, 1:3] *= 0
        """Z Normalization"""
        motion = (motion - self.mean) / self.std

        left_over_cnt = len(motion) % 8
        # print(motion.shape)
        if left_over_cnt != 0:
            if random.random() > 0.5:
                motion = motion[left_over_cnt:]
            else:
                motion = motion[:-left_over_cnt]
        # print(motion.shape)

        action_id = 0
        action_id = int(name.split("#")[-2])
        action_one_hot = np.zeros(self.opt.num_of_action)
        action_one_hot[action_id] = 1

        return motion, action_one_hot, action_id


class MotionBfaXiaEvalDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_xia_file_path, split_bfa_file_path):
        super().__init__()
        self.xia_dataset = MotionXiaDataset(opt, mean, std, split_xia_file_path, fix_bias=True)
        self.bfa_dataset = MotionDataset(opt, mean, std, split_bfa_file_path, fix_bias=True)
        self.len_bfa = len(self.bfa_dataset)

    def __len__(self):
        return len(self.xia_dataset)

    def inv_transform(self, input):
        return self.bfa_dataset.inv_transform(input)

    def __getitem__(self, item):
        mx, ax_oh, ax_id = self.xia_dataset[item]
        bfa_idx = random.randint(0, self.len_bfa-1)
        # print(bfa_idx, self.len_bfa)
        mb, _, _, _, _, sb_oh, sb_id, _, _ = self.bfa_dataset[bfa_idx]
        return mx, mb, ax_oh, ax_id, sb_oh, sb_id


class MotionBfaCMUTrainDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_cmu_file_path, split_bfa_file_path):
        super().__init__()
        self.opt = opt
        cmu_data = np.load(split_cmu_file_path, allow_pickle=True).item()
        bfa_data = np.load(split_bfa_file_path, allow_pickle=True).item()
        new_data = {}
        name_list = []
        style_dict = collections.defaultdict(list)
        sequence_dict = collections.defaultdict(list)

        for key, value in cmu_data.items():
            if len(value) < opt.motion_length:
                continue
            # key = "C_" + key
            new_data["C#" + key] = value
            name_list.append("C#" + key)
            # style_id = key.split("#")[-1]
            sequence_id = key.split("#")[0]
            if sequence_id.startswith("m_"):
                sequence_id = sequence_id[2:]
            sequence_id = "C_" + sequence_id
            # style_dict[style_id].append(key)
            sequence_dict[sequence_id].append("C#" + key)

        for key, value in bfa_data.items():
            if len(value) < opt.motion_length:
                continue
            # key = "B_" + key
            new_data["B#" + key] = value
            name_list.append("B#" + key)
            # style_id = key.split("#")[-1]
            sequence_id = key.split("#")[0]
            if sequence_id.startswith("m_"):
                sequence_id = sequence_id[2:]
            sequence_id = "B_" + sequence_id
            # style_dict[style_id].append(key)
            sequence_dict[sequence_id].append("B#" + key)

        if opt.is_train:
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4: 4 + opt.joint_num * 3] = std[4: 4 + opt.joint_num * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + opt.joint_num * 3: 4 + opt.joint_num * 9] = std[4 + opt.joint_num * 3:
                                                                      4 + opt.joint_num * 9] / 1.0
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + opt.joint_num * 9:
                4 + opt.joint_num * 12] = std[4 + opt.joint_num * 9:
                                               4 + opt.joint_num * 12] / 1.0
            # foot contact (B, seq_len, 4)
            std[4 + opt.joint_num * 12:] = std[4 + opt.joint_num * 12:] / opt.feat_bias

            assert 4 + opt.joint_num * 12 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
            np.save(pjoin(opt.meta_dir, 'std.npy'), std)

        self.mean = mean
        self.std = std
        self.data_dict = new_data
        self.name_list = name_list
        self.style_dict = style_dict
        self.sequence_dict = sequence_dict

    def __len__(self):
        return len(self.name_list)

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __getitem__(self, item):
        name = self.name_list[item]
        # style_id = self.style_dict[name]
        dataset_name = name.split("#")[0]
        sequence_id = name.split("#")[1]
        sequence_id = sequence_id[2:] if sequence_id.startswith("m_") else sequence_id
        sequence_id = "%s_%s"%(dataset_name, sequence_id)

        motion = self.data_dict[name]
        # Motion from the same sequence
        # print(sequence_id)
        # print(self.sequence_dict[sequence_id])
        motion1 = self.data_dict[random.choice([n for n in self.sequence_dict[sequence_id] if n!=name])]

        # Mask out root velecity information
        # motion[:, 1:3] *= 0
        """Z Normalization"""
        motion = (motion - self.mean) / self.std
        motion1 = (motion1 - self.mean) / self.std

        # print("After", motion[0])

        # m_length = data.shape[0]
        assert len(motion) >= self.opt.motion_length
        idx = random.randint(0, len(motion) - self.opt.motion_length)
        idx1 = random.randint(0, len(motion) - self.opt.motion_length)

        data = motion[idx:idx + self.opt.motion_length]
        # Motion from the same sequence
        data1 = motion1[idx1:idx1 + self.opt.motion_length]

        # data = motion
        # # Motion from the same sequence
        # data1 = motion1


        style_id = int(name.split("#")[-1])
        # BFA dataset does not contain action information
        action_id = 0
        if self.opt.dataset_name == "xia":
            action_id = int(name.split("#")[-2])

        style_one_hot = np.zeros(self.opt.num_of_style)
        style_one_hot[style_id] = 1

        action_one_hot = np.zeros(self.opt.num_of_action)
        action_one_hot[action_id] = 1

        return data, data1, action_one_hot, style_one_hot, style_id


class MotionBfaCMUEvalDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_cmu_file_path, split_bfa_file_path):
        super().__init__()
        self.opt = opt
        cmu_data = np.load(split_cmu_file_path, allow_pickle=True).item()
        bfa_data = np.load(split_bfa_file_path, allow_pickle=True).item()
        cmu_new_data = {}
        cmu_name_list = []
        bfa_new_data = {}
        bfa_name_list = []


        for key, value in cmu_data.items():
            if len(value) < opt.motion_length:
                continue
            # key = "C_" + key
            cmu_new_data[key] = value
            cmu_name_list.append(key)

        for key, value in bfa_data.items():
            if len(value) < opt.motion_length:
                continue
            # key = "B_" + key
            bfa_new_data[key] = value
            bfa_name_list.append(key)


        self.mean = mean
        self.std = std
        self.cmu_data_dict = cmu_new_data
        self.bfa_data_dict = bfa_new_data
        self.cmu_name_list = cmu_name_list
        self.bfa_name_list = bfa_name_list

    def __len__(self):
        return len(self.cmu_name_list)

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __getitem__(self, item):
        cmu_name = self.cmu_name_list[item]
        # style_id = self.style_dict[name]
        bfa_idx = random.randint(0, len(self.bfa_name_list) - 1)
        bfa_name = self.bfa_name_list[bfa_idx]

        cmu_motion = self.cmu_data_dict[cmu_name]
        bfa_motion = self.bfa_data_dict[bfa_name]
        # Mask out root velecity information
        # motion[:, 1:3] *= 0
        """Z Normalization"""
        cmu_motion = (cmu_motion - self.mean) / self.std
        bfa_motion = (bfa_motion - self.mean) / self.std

        # print("After", motion[0])

        # m_length = data.shape[0]
        assert min(len(cmu_motion), len(bfa_motion)) >= self.opt.motion_length
        cmu_idx = random.randint(0, len(cmu_motion) - self.opt.motion_length)
        bfa_idx = random.randint(0, len(bfa_motion) - self.opt.motion_length)

        cmu_data = cmu_motion[cmu_idx:cmu_idx + self.opt.motion_length]
        # Motion from the same sequence
        bfa_data = bfa_motion[bfa_idx:bfa_idx + self.opt.motion_length]

        # data = motion
        # # Motion from the same sequence
        # data1 = motion1

        style_id = int(bfa_name.split("#")[-1])
        # BFA dataset does not contain action information
        action_id = 0

        style_one_hot = np.zeros(self.opt.num_of_style)
        style_one_hot[style_id] = 1

        action_one_hot = np.zeros(self.opt.num_of_action)
        action_one_hot[action_id] = 1

        return cmu_data, bfa_data, action_one_hot, action_id, style_one_hot, style_id
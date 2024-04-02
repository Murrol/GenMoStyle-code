import os
import random
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class MotionDataset(Dataset):
    def __init__(self, phase, config, transform=None, ):
        super(MotionDataset, self).__init__()
        self.joint_num = 21
        self.config = config
        self.device = config["device"]
        data_dir = config["data_dir"]
        if phase == 'train':
            data_npy_path = os.path.join(data_dir, 'train_data.npy')
        else:
            data_npy_path = os.path.join(data_dir, 'test_data.npy')

        mdataset = np.load(data_npy_path, allow_pickle=True).item()
        self.motions, self.labels, self.actions, self.uids = [], [], [], []
        for key, value in mdataset.items():
            # print(len(value), self.config['motion_length'])
            if len(value)>=self.config['motion_length']:
                # if "cmu" not in config["data_dir"]:
                #     if eval(key.split("#")[-1]) != config["styleid"]:
                #         continue
                # print(len(value))
                self.uids.append(key)
                self.motions.append(value)
                if "cmu" in config["data_dir"]:
                    self.labels.append(0)
                else:
                    self.labels.append(eval(key.split("#")[-1]))

                if "xia" in config["data_dir"]:
                    self.actions.append(eval(key.split("#")[-2]))
                else:
                    self.actions.append(0)

        # data_norm_dir = os.path.join(data_dir, './') #TODO This ugly hard code.
        data_norm_dir = "../../motion_transfer_data/processed_bfa"
        motion_mean_path = os.path.join(data_norm_dir, "Mean.npy")
        motion_std_path = os.path.join(data_norm_dir, "Std.npy")
        # root_mean_path = os.path.join(data_norm_dir, "root_mean.npy")
        # root_std_path = os.path.join(data_norm_dir, "root_std.npy")
        if os.path.exists(motion_mean_path) and os.path.exists(motion_std_path):
            self.motion_mean = torch.from_numpy(np.load(motion_mean_path, allow_pickle=True).astype(np.float32)).to(self.device)
            self.motion_std = torch.from_numpy(np.load(motion_std_path, allow_pickle=True).astype(np.float32)).to(self.device)
            # self.root_mean = np.load(root_mean_path, allow_pickle=True).astype(np.float32)
            # self.root_std = np.load(root_std_path, allow_pickle=True).astype(np.float32)
        else:
            assert self.motion_mean and self.motion_std, 'no motion_mean or no motion_std'

        self.transform = transform

    def __len__(self):
        return len(self.motions)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        motion_raw = self.motions[index].astype(np.float32)
        label = self.labels[index]
        action = self.actions[index]
        uid = self.uids[index]
        
        # motion = np.transpose(motion_raw, (2, 1, 0)) # (seq, joint, dim) -> (dim, joint, seq)
        motion = torch.from_numpy(motion_raw).to(self.device)

        assert len(motion) >= self.config['motion_length']
        if self.config['motion_length'] > 20:
            roundlen = self.config['motion_length']
        else:
            roundlen = len(motion) // 16 * 16
        idx = random.randint(0, len(motion) - roundlen)
        motion = motion[idx:idx+roundlen]
        uid = uid + f"_{idx}_{roundlen}"

        root_data_raw = motion[:, :4].detach()
        root_data_raw = torch.permute(root_data_raw, (1, 0))
        positions = motion[:, 4:4 + self.joint_num * 3]
        rotations = motion[:, 4 + self.joint_num * 3 : 4 + self.joint_num * 9]
        velocities = motion[:, 4 + self.joint_num * 9 : 4 + self.joint_num * 12]
        foot_contact = motion[:, 4 + self.joint_num * 12:]
        positions = positions.reshape(-1, self.joint_num, 3)
        rotations = rotations.reshape(-1, self.joint_num, 6)
        velocities = velocities.reshape(-1, self.joint_num, 3)

        motion_raw = torch.concat([positions, rotations, velocities], axis=-1).detach()
        motion_raw = torch.permute(motion_raw, (2, 1, 0))
        # trans_p = float(np.random.rand(1))
        # if self.transform and trans_p < 0.2:
        #     motion = self.transform(motion)

        motion = (motion - self.motion_mean) \
                    / self.motion_std   # normalization

        root_data = motion[:, :4].detach()
        root_data = torch.permute(root_data, (1, 0))
        positions = motion[:, 4:4 + self.joint_num * 3]
        rotations = motion[:, 4 + self.joint_num * 3 : 4 + self.joint_num * 9]
        velocities = motion[:, 4 + self.joint_num * 9 : 4 + self.joint_num * 12]
        foot_contact = motion[:, 4 + self.joint_num * 12:]
        positions = positions.reshape(-1, self.joint_num, 3)
        rotations = rotations.reshape(-1, self.joint_num, 6)
        velocities = velocities.reshape(-1, self.joint_num, 3)

        motion = torch.concat([positions, rotations, velocities], axis=-1).detach()
        motion = torch.permute(motion, (2, 1, 0))

        data = {
                "motion_raw": motion_raw,
                "motion": motion,
                "root_raw": root_data_raw,
                "root": root_data,
                "foot_contact": foot_contact,
                "label": label,
                "action": action,
                "uid": uid,
                }

        return data

    def indexing(self, uid, idx=-1):
        index = self.uids.index(uid)
        motion_raw = self.motions[index].astype(np.float32)
        label = self.labels[index]
        action = self.actions[index]
        
        # motion = np.transpose(motion_raw, (2, 1, 0)) # (seq, joint, dim) -> (dim, joint, seq)
        motion = torch.from_numpy(motion_raw).to(self.device)
        
        
        assert len(motion) >= self.config['motion_length']
        if self.config['motion_length'] > 20:
            roundlen = self.config['motion_length']
        else:
            roundlen = len(motion) // 16 * 16

        if idx == -1:
            idx = random.randint(0, len(motion) - roundlen)

        motion = motion[idx:idx+roundlen]
        uid = uid + f"_{idx}_{roundlen}"

        root_data_raw = motion[:, :4].detach()
        root_data_raw = torch.permute(root_data_raw, (1, 0))
        positions = motion[:, 4:4 + self.joint_num * 3]
        rotations = motion[:, 4 + self.joint_num * 3 : 4 + self.joint_num * 9]
        velocities = motion[:, 4 + self.joint_num * 9 : 4 + self.joint_num * 12]
        foot_contact = motion[:, 4 + self.joint_num * 12:]
        positions = positions.reshape(-1, self.joint_num, 3)
        rotations = rotations.reshape(-1, self.joint_num, 6)
        velocities = velocities.reshape(-1, self.joint_num, 3)

        motion_raw = torch.concat([positions, rotations, velocities], axis=-1).detach()
        motion_raw = torch.permute(motion_raw, (2, 1, 0))
        # trans_p = float(np.random.rand(1))
        # if self.transform and trans_p < 0.2:
        #     motion = self.transform(motion)

        motion = (motion - self.motion_mean) \
                    / self.motion_std   # normalization

        root_data = motion[:, :4].detach()
        root_data = torch.permute(root_data, (1, 0))
        positions = motion[:, 4:4 + self.joint_num * 3]
        rotations = motion[:, 4 + self.joint_num * 3 : 4 + self.joint_num * 9]
        velocities = motion[:, 4 + self.joint_num * 9 : 4 + self.joint_num * 12]
        foot_contact = motion[:, 4 + self.joint_num * 12:]
        positions = positions.reshape(-1, self.joint_num, 3)
        rotations = rotations.reshape(-1, self.joint_num, 6)
        velocities = velocities.reshape(-1, self.joint_num, 3)

        motion = torch.concat([positions, rotations, velocities], axis=-1).detach()
        motion = torch.permute(motion, (2, 1, 0))

        data = {
                "motion_raw": motion_raw,
                "motion": motion,
                "root_raw": root_data_raw,
                "root": root_data,
                "foot_contact": foot_contact,
                "label": label,
                "action": action,
                "uid": uid,
                }

        return data



if __name__ == '__main__':
    import sys
    from etc.utils import print_composite
    sys.path.append('./motion')
    sys.path.append('./etc')
    from viz_motion import animation_plot    # for checking dataloader
    data_dir = './datasets/cmu/'
    
    batch_size = 2
    dataset = MotionDataset('train', data_dir)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for batch in data_loader:
        print_composite(batch)
        motion_raw = batch['motion_raw'].cpu().numpy()
        root_raw = batch['root_raw'].cpu().numpy()
        foot_contact = batch['foot_contact'].cpu().numpy()
        anim1 = [motion_raw[0], root_raw[0], foot_contact[0]]
        anim2 = [motion_raw[1], root_raw[1], foot_contact[1]]
        animation_plot([anim1, anim2])

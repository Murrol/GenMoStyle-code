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
        self.motions, self.labels, self.actions = [], [], []
        for key, value in mdataset.items():
            # print(len(value), self.config['motion_length'])
            if len(value)>=self.config['motion_length']:
                # print(len(value))
                self.motions.append(value)
                if "cmu" in config["data_dir"]:
                    self.labels.append(0)
                else:
                    self.labels.append(eval(key.split("#")[-1]))
                if "xia" in config["data_dir"]:
                    self.actions.append(eval(key.split("#")[-2]))
                else:
                    self.actions.append(0)
        # for temp in self.motions:
        #     print(temp)
        #     break
        # self.motions = mdataset["motion"]
        # self.roots = mdataset['root']
        # self.foot_contacts = mdataset["foot_contact"]
        
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

        if "cmu" in self.config["data_dir"]:
            diff_label_idx = index
        else:
            diff_label_idx = random.choice([idx for idx, lab in enumerate(self.labels) if lab != label])
        diff_label = self.labels[diff_label_idx]
        diff_action = self.actions[diff_label_idx]
        diff_motion = self.motions[diff_label_idx].astype(np.float32)
        
        # motion = np.transpose(motion_raw, (2, 1, 0)) # (seq, joint, dim) -> (dim, joint, seq)
        motion = torch.from_numpy(motion_raw).to(self.device)

        assert len(motion) >= self.config['motion_length']
        if self.config['motion_length'] > 20:
            idx = random.randint(0, len(motion) - self.config['motion_length'])
            motion = motion[idx:idx+self.config['motion_length']]
        else:
            roundlen = len(motion) // 16 * 16
            idx = random.randint(0, len(motion) - roundlen)
            motion = motion[idx:idx+roundlen]

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
        # root_raw = self.roots[index].astype(np.float32)
        # root = np.transpose(root_raw, (2, 1, 0)) # (seq, joint, dim) -> (dim, joint, seq)
        # root = torch.from_numpy(root)
        # root = (root - self.root_mean[:, np.newaxis, np.newaxis]) \
        #             / self.root_std[:, np.newaxis, np.newaxis]   # normalization

        # foot_contact = self.foot_contacts[index].astype(np.float32)

        diff_motion = torch.from_numpy(diff_motion).to(self.device)

        assert len(diff_motion) >= self.config['motion_length']
        if self.config['motion_length'] > 20:
            idx = random.randint(0, len(diff_motion) - self.config['motion_length'])
            diff_motion = diff_motion[idx:idx+self.config['motion_length']]
        else:
            roundlen = len(motion) // 16 * 16
            idx = random.randint(0, len(motion) - roundlen)
            diff_motion = diff_motion[idx:idx+roundlen]

        diff_root_data_raw = diff_motion[:, :4].detach()
        diff_root_data_raw = torch.permute(diff_root_data_raw, (1, 0))
        positions = diff_motion[:, 4:4 + self.joint_num * 3]
        rotations = diff_motion[:, 4 + self.joint_num * 3 : 4 + self.joint_num * 9]
        velocities = diff_motion[:, 4 + self.joint_num * 9 : 4 + self.joint_num * 12]
        diff_foot_contact = diff_motion[:, 4 + self.joint_num * 12:]
        positions = positions.reshape(-1, self.joint_num, 3)
        rotations = rotations.reshape(-1, self.joint_num, 6)
        velocities = velocities.reshape(-1, self.joint_num, 3)

        diff_motion_raw= torch.concat([positions, rotations, velocities], axis=-1).detach()
        diff_motion_raw = torch.permute(diff_motion_raw, (2, 1, 0))
        # trans_p = float(np.random.rand(1))
        # if self.transform and trans_p < 0.2:
        #     motion = self.transform(motion)

        diff_motion = (diff_motion - self.motion_mean) \
                    / self.motion_std   # normalization

        diff_root_data = diff_motion[:, :4].detach()
        diff_root_data = torch.permute(diff_root_data, (1, 0))
        positions = diff_motion[:, 4:4 + self.joint_num * 3]
        rotations = diff_motion[:, 4 + self.joint_num * 3 : 4 + self.joint_num * 9]
        velocities = diff_motion[:, 4 + self.joint_num * 9 : 4 + self.joint_num * 12]
        diff_foot_contact = diff_motion[:, 4 + self.joint_num * 12:]
        positions = positions.reshape(-1, self.joint_num, 3)
        rotations = rotations.reshape(-1, self.joint_num, 6)
        velocities = velocities.reshape(-1, self.joint_num, 3)

        diff_motion = torch.concat([positions, rotations, velocities], axis=-1).detach()
        diff_motion = torch.permute(diff_motion, (2, 1, 0))

        data = {
                "motion_raw": motion_raw,
                "motion": motion,
                "root_raw": root_data_raw,
                "root": root_data,
                "foot_contact": foot_contact,
                "label": label,
                "action": action,

                "diff_motion_raw": diff_motion_raw,
                "diff_motion": diff_motion,
                "diff_root_raw": diff_root_data_raw,
                "diff_root": diff_root_data,
                # "foot_contact": foot_contact
                "diff_label": diff_label,
                "diff_action": diff_action,
                }

        return data


class RandomResizedCrop(object):
    """Crop and resize randomly the motion in a sample."""
    def __call__(self, sample):
        global crop
        c, j, s = sample.shape      # (dim, joint, seq)

        idx = random.randint(30, 90)
        size = random.randint(60, 120)

        if idx > (size//2)+(size%2) and idx+(size//2) < 120:
            crop = sample[..., idx-(size//2)-(size%2):idx+(size//2)]
        elif idx <= (size//2)+(size%2):
            crop = sample[..., :idx+(size//2)]
        elif  idx+(size//2) >= 120:
            crop = sample[..., idx-(size//2)-(size%2):]

        if size < 90:
            scale = random.uniform(1, 2)
        else:
            scale = random.uniform(0.5, 1)
        crop = crop.unsqueeze(0)
        # crop = torch.from_numpy(crop).unsqueeze(0)
        scale_crop = F.interpolate(crop, scale_factor=(1, scale), mode='bilinear',
                                   align_corners=True, recompute_scale_factor=True)
        scale_crop = scale_crop.squeeze(0)

        if scale_crop.shape[-1] > 120:
            scale_crop = scale_crop[..., scale_crop.shape[-1]//2-60:scale_crop.shape[-1]//2+60]
            return scale_crop
        else:
            # padding
            left = scale_crop[..., :1].repeat_interleave(
                (120-scale_crop.shape[-1])//2 + (120-scale_crop.shape[-1]) % 2, dim=-1)
            left[-3:] = 0.0
            right = scale_crop[..., -1:].repeat_interleave((120-scale_crop.shape[-1])//2, dim=-1)
            right[-3:] = 0.0
            padding_scale_crop = torch.cat([left, scale_crop, right], dim=-1)

        return padding_scale_crop


def get_dataloader(subset_name, config, seed=None, shuffle=None, transform=None, drop_last=False):
    dataset = MotionDataset(subset_name, config, transform)
    batch_size = config['batch_size'] #if subset_name == 'train' else 1  # since dataloader
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=(subset_name == 'train') if shuffle is None else shuffle,
                      num_workers=config['num_workers'] if subset_name == 'train' else 0,
                    #   worker_init_fn=np.random.seed(seed) if seed else None,
                    #   pin_memory=True, #cannot run on cuda
                      drop_last=drop_last)


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

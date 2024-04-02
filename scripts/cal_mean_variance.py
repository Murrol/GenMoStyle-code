import numpy as np
import sys
import os
from os.path import join as pjoin


# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, joint_num * 3)
# rot_data (B, seq_len, joint_num * 6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
def mean_variance(data_dir, save_dir, joints_num):
    file_list = os.listdir(data_dir)
    data_list = []

    for file in file_list:
        data = np.load(pjoin(data_dir, file))
        if np.isnan(data).any():
            print(file)
            continue
        data_list.append(data)

    data = np.concatenate(data_list, axis=0)
    print(data.shape)
    Mean = data.mean(axis=0)
    Std = data.std(axis=0)
    Std[0:1] = Std[0:1].mean() / 1.0
    Std[1:3] = Std[1:3].mean() / 1.0
    Std[3:4] = Std[3:4].mean() / 1.0
    Std[4: 4+joints_num * 3] = Std[4: 4+joints_num * 3].mean() / 1.0
    Std[4+joints_num * 3: 4+joints_num * 9] = Std[4+joints_num * 3: 4+joints_num * 9].mean() / 1.0
    Std[4+joints_num * 9: 4+joints_num * 9 + joints_num*3] = Std[4+joints_num * 9: 4+joints_num * 9 + joints_num*3].mean() / 1.0
    Std[4 + joints_num * 9 + joints_num * 3: ] = Std[4 + joints_num * 9 + joints_num * 3: ].mean() / 1.0

    assert 8 + joints_num * 12 == Std.shape[-1]

    np.save(pjoin(save_dir, 'Mean.npy'), Mean)
    np.save(pjoin(save_dir, 'Std.npy'), Std)

    return Mean, Std

def mean_variance_2(data_dir1, data_dir2, save_dir, joints_num):
    file_list1 = os.listdir(data_dir1)
    data_list = []

    for file in file_list1:
        data = np.load(pjoin(data_dir1, file))
        if np.isnan(data).any():
            print(file)
            continue
        data_list.append(data)

    file_list2 = os.listdir(data_dir2)

    for file in file_list2:
        data = np.load(pjoin(data_dir2, file))
        if np.isnan(data).any():
            print(file)
            continue
        data_list.append(data)

    data = np.concatenate(data_list, axis=0)
    print(data.shape)
    Mean = data.mean(axis=0)
    Std = data.std(axis=0)
    Std[0:1] = Std[0:1].mean() / 1.0
    Std[1:3] = Std[1:3].mean() / 1.0
    Std[3:4] = Std[3:4].mean() / 1.0
    Std[4: 4+joints_num * 3] = Std[4: 4+joints_num * 3].mean() / 1.0
    Std[4+joints_num * 3: 4+joints_num * 9] = Std[4+joints_num * 3: 4+joints_num * 9].mean() / 1.0
    Std[4+joints_num * 9: 4+joints_num * 9 + joints_num*3] = Std[4+joints_num * 9: 4+joints_num * 9 + joints_num*3].mean() / 1.0
    Std[4 + joints_num * 9 + joints_num * 3: ] = Std[4 + joints_num * 9 + joints_num * 3: ].mean() / 1.0

    assert 8 + joints_num * 12 == Std.shape[-1]

    np.save(pjoin(save_dir, 'Mean.npy'), Mean)
    np.save(pjoin(save_dir, 'Std.npy'), Std)

    return Mean, Std


if __name__ == '__main__':
    data_dir1 = '../../data/motion_transfer/processed_cmu/npy/'
    data_dir2 = '../../data/motion_transfer/processed_bfa/npy/'
    save_dir = '../../data/motion_transfer/processed_cmu/'
    mean, Std = mean_variance_2(data_dir1, data_dir2, save_dir, 21)
    print(mean)
    print(Std)
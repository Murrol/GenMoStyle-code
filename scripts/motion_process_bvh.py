from os.path import join as pjoin
import scipy.ndimage.filters as filters

from common.skeleton1 import Skeleton
import numpy as np
import os
import random
from common.quaternion import *
from common.paramUtil import *

import torch
from tqdm import tqdm
import motion.BVH as BVH
from common.paramUtil import *
import pickle

remain_joints = [0, 2, 3, 4, 5,
                  7,  8,  9, 10,
                 12, 13, 15, 16,
                 18, 19, 20, 22,
                 25, 26, 27, 29]

fid_l, fid_r = [3, 4], [7, 8]


def process_data(animation, ds_rate, face_joint_idx, feet_thre):

    """Downsampling to fps"""
    rotations = animation.rotations[::ds_rate]
    positions = animation.positions[::ds_rate]

    """Do FK"""
    skeleton = Skeleton(animation.offsets, animation.parents, "cpu")
    global_quat, global_pos = skeleton.fk_local_quat_np(rotations, positions[:, 0])

    """Remove unuseful joints"""
    global_quat = global_quat[:, np.array(remain_joints)]
    global_pos = global_pos[:, np.array(remain_joints)]

    """Put on Floor"""
    floor_height = global_pos.min(axis=0).min(axis=0)[1]
    global_pos[:, :, 1] -= floor_height

    """XZ at Origin"""
    root_pos_init_xz = global_pos[0, 0] * np.array([1, 0, 1])
    global_pos = global_pos - root_pos_init_xz

    """Extract Forward Direction and Smooth"""
    r_hip, l_hip, r_sdr, l_sdr = face_joint_idx
    # across1 = root_pos_init[]
    across = (
        (global_pos[:, l_sdr] - global_pos[:, r_sdr]) +
        (global_pos[:, l_hip] - global_pos[:, r_hip])
    )
    across = across / np.sqrt((across**2).sum(axis=-1)[..., np.newaxis])
    direction_filterwidth = 10
    forward = filters.gaussian_filter1d(
        np.cross(across, np.array([[0, 1, 0]])), direction_filterwidth, axis=0, mode="nearest")

    forward = forward / np.sqrt((forward**2).sum(axis=-1)[..., np.newaxis])
    target = np.array([[0, 0, 1]]).repeat(len(forward), axis=0)
    root_rotations = qbetween_np(forward, target)[:, np.newaxis]
    root_rotations = np.repeat(root_rotations, global_pos.shape[1], axis=1)

    """All Initially Face Z+"""
    root_rotation_init = root_rotations[0:1].repeat(len(root_rotations), axis=0)
    root_rotations = qmul_np(qinv_np(root_rotation_init), root_rotations)
    global_quat = qmul_np(root_rotation_init, global_quat)
    global_pos = qrot_np(root_rotation_init, global_pos)

    """Re-gain Global Positions"""
    # Due to the change of skeletons and rotations, the global_quat can not be precisely converted to global_pos now,
    # although they are very close.
    skeleton = Skeleton(animation.offsets[remain_joints], parents, "cpu")
    global_pos = skeleton.fk_global_quat_np(global_quat, global_pos[:, 0])

    """ Get Foot Contacts """
    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        feet_l_h = positions[:-1, fid_l, 1]
        feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
        #     feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        feet_r_h = positions[:-1, fid_r, 1]
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
        #     feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float)
        return feet_l, feet_r

    def get_cont6d_params(r_rot, r_pos, quat_params):
        """remove root rotations from joint rotations"""
        quat_params = qmul_np(r_rot, quat_params)

        """Quaternion to Continuous 6D"""
        cont6d_params = quat_to_cont6d_np(quat_params)

        """Root Linear Velocity"""
        velocity = (r_pos[1:] - r_pos[:-1]).copy()
        velocity = qrot_np(r_rot[:-1, 0], velocity)

        """Root Angular Velocity"""
        r_velocity = qmul_np(r_rot[1:, 0], qinv_np(r_rot[:-1, 0]))
        #     print(r_velocity.shape)
        r_velocity = np.arcsin(r_velocity[:, 2:3]) * 2
        #     print(r_velocity.shape)
        return cont6d_params[:-1], velocity, r_velocity

    def get_local_positions(r_rot, positions):
        positions = positions.copy()
        '''Local pose'''
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        '''All pose face Z+'''
        positions = qrot_np(r_rot, positions)

        '''Get Joint Velocity'''
        # (seq_len-1, joints_num*3)
        local_vel = positions[1:] - positions[:-1]
        return positions[:-1], local_vel

    feet_l, feet_r = foot_detect(global_pos, thres=feet_thre)

    cont6d_params, l_velocity, r_velocity = get_cont6d_params(root_rotations, global_pos[:, 0], global_quat)

    local_positions, local_velocity = get_local_positions(root_rotations, global_pos)

    """Root height"""
    root_y = local_positions[:, 0, 1:2]

    """Linear Root Velocity"""
    l_velocity = l_velocity[:, [0, 2]]

    """Get Root Data"""
    root_data = np.concatenate([r_velocity, l_velocity, root_y], axis=-1)

    """Get Joint Rotation Representation"""
    rot_data = cont6d_params.reshape(len(cont6d_params), -1)

    """Get Joint Rotation Invariant Position Representation"""
    ric_data = local_positions.reshape(len(local_positions), -1)

    """Get Joint Velocity Representation"""
    vel_data = local_velocity.reshape(len(local_velocity), -1)

    data = np.concatenate([root_data, ric_data, rot_data, vel_data, feet_l, feet_r], axis=-1)
    return data

def process_data_full(animation, ds_rate, face_joint_idx, feet_thre):

    """Downsampling to fps"""
    rotations = animation.rotations[::ds_rate]
    positions = animation.positions[::ds_rate]

    """Do FK"""
    # print(animation.offsets.shape)
    # print(animation.parents)
    skeleton = Skeleton(animation.offsets, animation.parents, "cpu")
    global_quat, global_pos = skeleton.fk_local_quat_np(rotations, positions[:, 0])

    """Remove unuseful joints"""
    global_quat = global_quat
    global_pos = global_pos

    """Put on Floor"""
    floor_height = global_pos.min(axis=0).min(axis=0)[1]
    global_pos[:, :, 1] -= floor_height

    """XZ at Origin"""
    root_pos_init_xz = global_pos[0, 0] * np.array([1, 0, 1])
    global_pos = global_pos - root_pos_init_xz

    """Extract Forward Direction and Smooth"""
    r_hip, l_hip, r_sdr, l_sdr = face_joint_idx
    # across1 = root_pos_init[]
    across = (
        (global_pos[:, l_sdr] - global_pos[:, r_sdr]) +
        (global_pos[:, l_hip] - global_pos[:, r_hip])
    )
    across = across / np.sqrt((across**2).sum(axis=-1)[..., np.newaxis])
    direction_filterwidth = 10
    forward = filters.gaussian_filter1d(
        np.cross(across, np.array([[0, 1, 0]])), direction_filterwidth, axis=0, mode="nearest")

    forward = forward / np.sqrt((forward**2).sum(axis=-1)[..., np.newaxis])
    target = np.array([[0, 0, 1]]).repeat(len(forward), axis=0)
    root_rotations = qbetween_np(forward, target)[:, np.newaxis]
    root_rotations = np.repeat(root_rotations, global_pos.shape[1], axis=1)

    """All Initially Face Z+"""
    root_rotation_init = root_rotations[0:1].repeat(len(root_rotations), axis=0)
    root_rotations = qmul_np(qinv_np(root_rotation_init), root_rotations)
    global_quat = qmul_np(root_rotation_init, global_quat)
    global_pos = qrot_np(root_rotation_init, global_pos)

    """Re-gain Global Positions"""
    # Due to the change of skeletons and rotations, the global_quat can not be precisely converted to global_pos now,
    # although they are very close.
    skeleton = Skeleton(animation.offsets, parents, "cpu")
    global_pos = skeleton.fk_global_quat_np(global_quat, global_pos[:, 0])

    """ Get Foot Contacts """
    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        feet_l_h = positions[:-1, fid_l, 1]
        feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
        #     feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        feet_r_h = positions[:-1, fid_r, 1]
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
        #     feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float)
        return feet_l, feet_r

    def get_cont6d_params(r_rot, r_pos, quat_params):
        """remove root rotations from joint rotations"""
        quat_params = qmul_np(r_rot, quat_params)

        """Quaternion to Continuous 6D"""
        cont6d_params = quat_to_cont6d_np(quat_params)

        """Root Linear Velocity"""
        velocity = (r_pos[1:] - r_pos[:-1]).copy()
        velocity = qrot_np(r_rot[:-1, 0], velocity)

        """Root Angular Velocity"""
        r_velocity = qmul_np(r_rot[1:, 0], qinv_np(r_rot[:-1, 0]))
        #     print(r_velocity.shape)
        r_velocity = np.arcsin(r_velocity[:, 2:3]) * 2
        #     print(r_velocity.shape)
        return cont6d_params[:-1], velocity, r_velocity

    def get_local_positions(r_rot, positions):
        positions = positions.copy()
        '''Local pose'''
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        '''All pose face Z+'''
        positions = qrot_np(r_rot, positions)

        '''Get Joint Velocity'''
        # (seq_len-1, joints_num*3)
        local_vel = positions[1:] - positions[:-1]
        return positions[:-1], local_vel

    feet_l, feet_r = foot_detect(global_pos, thres=feet_thre)

    cont6d_params, l_velocity, r_velocity = get_cont6d_params(root_rotations, global_pos[:, 0], global_quat)

    local_positions, local_velocity = get_local_positions(root_rotations, global_pos)

    """Root height"""
    root_y = local_positions[:, 0, 1:2]

    """Linear Root Velocity"""
    l_velocity = l_velocity[:, [0, 2]]

    """Get Root Data"""
    root_data = np.concatenate([r_velocity, l_velocity, root_y], axis=-1)

    """Get Joint Rotation Representation"""
    rot_data = cont6d_params.reshape(len(cont6d_params), -1)

    """Get Joint Rotation Invariant Position Representation"""
    ric_data = local_positions.reshape(len(local_positions), -1)

    """Get Joint Velocity Representation"""
    vel_data = local_velocity.reshape(len(local_velocity), -1)

    data = np.concatenate([root_data, ric_data, rot_data, vel_data, feet_l, feet_r], axis=-1)
    return data

# Recover global angle and positions for rotation data
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, joint_num * 3)
# rot_data (B, seq_len, joint_num * 6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    """Get Y-axis Rotation from Rotation Velocity"""
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang / 2, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    # (vx, vy, vz, r) - > (cos(r/2), vx * sin(r/2), vy * sin(r/2), vz * sin(r/2))
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    """Get Root Positions"""
    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]

    #     print(torch.sum(r_pos**2, axis=-1)[:100])
    """Add Y-axis Rotation to Root Positions"""
    r_pos = qrot(qinv(r_rot_quat), r_pos)
    #     print(torch.sum(r_pos**2, axis=-1)[:100])

    r_pos = torch.cumsum(r_pos, dim=-2)
    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos

# Recover global angle and positions for rotation data
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, joint_num * 3)
# rot_data (B, seq_len, joint_num * 6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
def recover_bvh_from_rot(data, joints_num, skeleton):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    start_indx = 1 + 2 + 1 + joints_num * 3
    end_indx = start_indx + joints_num * 6
    cont6d_params = data[..., start_indx:end_indx].reshape(-1, joints_num, 6)
    quat_params = cont6d_to_quat(cont6d_params)
#     print(r_rot_quat.shape, quat_params.shape)
    global_quats = qmul(qinv(r_rot_quat)[:, np.newaxis].repeat(1,joints_num, 1), quat_params)
    local_quats = skeleton.global_to_local_quat(global_quats)
    return global_quats, local_quats, r_pos

def recover_pos_from_rot(data, joints_num, skeleton):
    global_quats, _, r_pos = recover_bvh_from_rot(data, joints_num, skeleton)
    global_pos = skeleton.fk_global_quat(global_quats, r_pos)
    return global_pos

def recover_pos_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:joints_num * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))
#     print(positions.shape)

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

#     '''Concate root and joints'''
#     positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions

def train_test_split_bfa(source_dir, save_dir, test_prob=0.1, window=200, window_step=100):
    fnames = os.listdir(source_dir)
    train_data = {}
    test_data = {}
    for name in tqdm(fnames):
        info = name.split("_")
        if info[0] == "m":
            continue
        style = name.split("_")[0]
        style_id = bfa_style_inv_enumerator[style]
        data = np.load(pjoin(source_dir, name))
        data_m = np.load(pjoin(source_dir, "m_"+name))
        i = 0
        while i < len(data):
            flip = random.random()
            is_test = False
            if i != 0 and flip <= test_prob:
                start = i + window_step
                end = start + window
                i = end
                is_test = True
            else:
                start = i
                end = i + window
                i += window_step
            start = min(start, len(data))
            end = min(end, len(data))
            if len(data) in (start, end):
                break
            # action_id = bfa_style_inv_enumerator[]
            id = "%s#%d#%d#%d"%(name[:-4], start, end, style_id)
            m_id = "m_%s#%d#%d#%d"%(name[:-4], start, end, style_id)
            if is_test:
                test_data[id] = data[start:end]
                test_data[m_id] = data_m[start:end]
            else:
                train_data[id] = data[start:end]
                train_data[m_id] = data_m[start:end]
    print("%d Clips in Training Set"%(len(train_data)))
    print("%d Clips in Testing Set"%(len(test_data)))
    np.save(pjoin(save_dir, "train_data.npy"), train_data)
    np.save(pjoin(save_dir, "test_data.npy"), test_data)

def train_test_split_xia(source_dir, save_dir, test_prob=0.1):
    fnames = os.listdir(source_dir)
    train_data = {}
    test_data = {}
    for name in tqdm(fnames):
        info = name.split("_")
        if info[0] == "m":
            continue
        info = name.split("_")
        style = info[0]
        style_id = xia_style_inv_enumerator[style]
        if len(info) == 4:
            action = info[1]+"_"+info[2]
        else:
            action = info[1]
        action_id = xia_action_inv_enumerator[action]
        data = np.load(pjoin(source_dir, name))
        data_m = np.load(pjoin(source_dir, "m_" + name))
        flip = random.random()
        is_test = (flip<=test_prob)
        id = "%s#%d#%d"%(name, action_id, style_id)
        m_id = "m_%s#%d#%d"%(name, action_id, style_id)
        if is_test:
            test_data[id] = data
            test_data[m_id] = data_m
        else:
            train_data[id] = data
            train_data[m_id] = data_m
    print("%d Clips in Training Set" % (len(train_data)))
    print("%d Clips in Testing Set" % (len(test_data)))
    np.save(pjoin(save_dir, "train_data.npy"), train_data)
    np.save(pjoin(save_dir, "test_data.npy"), test_data)

def train_test_split_cmu(source_dir, save_dir, test_prob=0.1, window=200, window_step=100):
    fnames = os.listdir(source_dir)
    train_data = {}
    test_data = {}
    for name in tqdm(fnames):
        info = name.split("_")
        if info[0] == "m":
            continue
        # style = name.split("_")[0]
        # style_id = bfa_style_inv_enumerator[style]
        style_id = 0
        data = np.load(pjoin(source_dir, name))
        data_m = np.load(pjoin(source_dir, "m_"+name))
        i = 0
        while i < len(data):
            flip = random.random()
            is_test = False
            if i != 0 and flip <= test_prob:
                start = i + window_step
                end = start + window
                i = end
                is_test = True
            else:
                start = i
                end = i + window
                i += window_step
            start = min(start, len(data))
            end = min(end, len(data))
            if len(data) in (start, end):
                break
            # action_id = bfa_style_inv_enumerator[]
            id = "%s#%d#%d#%d"%(name[:-4], start, end, style_id)
            m_id = "m_%s#%d#%d#%d"%(name[:-4], start, end, style_id)
            if is_test:
                test_data[id] = data[start:end]
                test_data[m_id] = data_m[start:end]
            else:
                train_data[id] = data[start:end]
                train_data[m_id] = data_m[start:end]
    print("%d Clips in Training Set"%(len(train_data)))
    print("%d Clips in Testing Set"%(len(test_data)))
    np.save(pjoin(save_dir, "train_data.npy"), train_data)
    np.save(pjoin(save_dir, "test_data.npy"), test_data)

if __name__ == "__main__":
    source_dir = "../../data/motion_transfer/mocap_cmu/"
    tgt_dir = "../../data/motion_transfer/processed_cmu/"
    npy_dir = pjoin(tgt_dir, "npy")
    bvh_dir = pjoin(tgt_dir, "bvh")
    remain_joints = [0, 2, 3, 4, 5,
                     7, 8, 9, 10,
                     12, 13, 15, 16,
                     18, 19, 20, 22,
                     25, 26, 27, 29]

    fid_l, fid_r = [3, 4], [7, 8]
    # r_hip, l_hip, r_sdr, l_sdr
    face_joint_idx = [5, 1, 17, 13]
    # kinematic_chain = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [10, 13, 14, 15, 16], [10, 17, 18, 19, 20]]
    joints_num = 21
    parents = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 10, 13, 14, 15, 10, 17, 18, 19]
    ds_rate = 4
    window = 300
    window_step = 200
    test_prob = 0.1
    # fps = 30
    # radius = 40
    fnames = os.listdir(source_dir)
    os.makedirs(npy_dir, exist_ok=True)
    os.makedirs(bvh_dir, exist_ok=True)
    # for name in tqdm(fnames):
    #     animation = BVH.load(pjoin(source_dir, name))
    #     skeleton = Skeleton(animation.offsets[remain_joints], parents, "cpu")
    #     writer = BVH.WriterWrapper(parents, animation.frametime * ds_rate, animation.offsets[remain_joints])
    #     # if len(animation) < 4:
    #     #     continue
    #     # print(len(animation))
    #     try:
    #         data = process_data(animation, 4, face_joint_idx, 0.05)
    #         _, local_quats, r_pos = recover_bvh_from_rot(torch.from_numpy(data).float(), joints_num, skeleton)
    #         writer.write(pjoin(bvh_dir, name), local_quats.numpy(), r_pos.numpy(), "zyx",
    #                      names=[animation.names[i] for i in remain_joints])
    #         np.save(pjoin(npy_dir, name[:-3] + "npy"), data)
    #     except:
    #         print(len(animation))
    # train_test_split_bfa(pjoin(tgt_dir, "npy"), tgt_dir, test_prob=test_prob, window=window, window_step=window_step)
    # train_test_split_xia(npy_dir, tgt_dir, test_prob)
    train_test_split_cmu(pjoin(tgt_dir, "npy"), tgt_dir, test_prob=test_prob, window=window, window_step=window_step)


import os
import sys
import numpy as np
import scipy.ndimage.filters as filters

sys.path.append('../')
from animation import BVH, Animation
from animation.Quaternions import Quaternions
from animation.Pivots import Pivots

njoints = 21
selected_joints = [0,
                   2, 3, 4, 5,
                   7, 8, 9, 10,
                   12, 13, 15, 16,
                   18, 19, 20, 22,
                   25, 26, 27, 29]

parents = [-1,
            0, 1, 2, 3,
            0, 5, 6, 7,
            0, 9, 10, 11,
            10, 13, 14, 15,
            10, 17, 18, 19]

f = open('contents.txt', 'r')
contents = [line.strip() for line in f.readlines()]

f = open('styles.txt', 'r')
styles = [line.strip() for line in f.readlines()]


def get_bvh_files(directory):
    return [os.path.join(directory, f) for f in sorted(list(os.listdir(directory)))
            if os.path.isfile(os.path.join(directory, f))
            and f.endswith('.bvh') and f != 'rest.bvh']


def feet_contact_from_positions(positions, fid_l=(3, 4), fid_r=(7, 8)):
    fid_l, fid_r = np.array(fid_l), np.array(fid_r)
    velfactor = np.array([0.05, 0.05])
    feet_contact = []
    for fid_index in [fid_l, fid_r]:
        foot_vel = (positions[1:, fid_index] - positions[:-1, fid_index]) ** 2
        foot_vel = np.sum(foot_vel, axis=-1)
        foot_contact = (foot_vel < velfactor).astype(np.float)
        feet_contact.append(foot_contact)
    feet_contact = np.concatenate(feet_contact, axis=-1)
    feet_contact = np.concatenate((feet_contact[0:1].copy(), feet_contact), axis=0)

    return feet_contact


def preprocess(filename, downsample=2, slice=True, window=64, window_step=32):
    anim, names, frametime = BVH.load(filename)
    anim = anim[::downsample]

    global_xforms = Animation.transforms_global(anim)
    global_positions = global_xforms[:,:,:3,3] / global_xforms[:,:,3:,3]
    global_rotations = Quaternions.from_transforms(global_xforms)

    global_positions = global_positions[:, selected_joints]
    global_rotations = global_rotations[:, selected_joints]

    clip, feet = get_motion_data(global_positions, global_rotations)

    if not slice:
        return clip, feet

    else:
        cls = np.array([-1, -1])
        clip_windows = []
        feet_windows = []
        class_windows = []

        cls_name = os.path.split(filename)[1]
        cls = np.array([contents.index(cls_name.split('_')[0].split()[-1]),
                        styles.index(cls_name.split('_')[1])])

        if not (cls[0] < 0) & (cls[1] < 0):
            for j in range(0, len(clip) - window // 8, window_step):
                assert (len(global_positions) >= window // 8)
                clip_slice = clip[j:j + window]
                clip_feet = feet[j:j + window]

                if len(clip_slice) < window:
                    # left slices
                    clip_left = clip_slice[:1].repeat((window - len(clip_slice)) // 2 + (window - len(clip_slice)) % 2, axis=0)
                    clip_left[:, :, -4:] = 0.0
                    clip_feet_l = clip_feet[:1].repeat((window - len(clip_slice)) // 2 + (window - len(clip_slice)) % 2, axis=0)
                    # right slices
                    clip_right = clip_slice[-1:].repeat((window - len(clip_slice)) // 2, axis=0)
                    clip_right[:, :, -4:] = 0.0
                    clip_feet_r = clip_feet[-1:].repeat((window - len(clip_slice)) // 2, axis=0)
                    # padding
                    clip_slice = np.concatenate([clip_left, clip_slice, clip_right], axis=0)
                    clip_feet = np.concatenate([clip_feet_l, clip_feet, clip_feet_r], axis=0)
                if len(clip_slice) != window: raise Exception()
                if len(clip_feet) != window: raise Exception()

                clip_windows.append(clip_slice)
                feet_windows.append(clip_feet)
                class_windows.append(cls)

        return clip_windows, feet_windows, class_windows


def get_motion_data(global_positions, global_rotations):
    # extract forward direction
    sdr_l, sdr_r, hip_l, hip_r = 13, 17, 1, 5
    across = ((global_positions[:, sdr_l] - global_positions[:, sdr_r]) + (global_positions[:, hip_l] - global_positions[:, hip_r]))
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]  # (F, 3)

    # smooth forward direction
    direction_filterwidth = 20
    forward = filters.gaussian_filter1d(np.cross(across, np.array([[0, 1, 0]])), direction_filterwidth, axis=0, mode='nearest')
    forward = forward / np.sqrt((forward ** 2).sum(axis=-1))[..., np.newaxis]

    # remove translation & rotation
    root_rotation = Quaternions.between(forward, np.array([[0, 0, 1]]).repeat(len(forward), axis=0))[:, np.newaxis]
    positions = global_positions.copy()
    rotations = global_rotations.copy()
    positions[:, :, 0] = positions[:, :, 0] - positions[:, 0:1, 0]
    positions[:, :, 1] = positions[:, :, 1] - positions[:, 0:1, 1] + positions[0:1, 0:1, 1]
    positions[:, :, 2] = positions[:, :, 2] - positions[:, 0:1, 2]
    positions = root_rotation * positions
    rotations = root_rotation * rotations

    # trajectory info
    root_velocity = root_rotation[:-1] * (global_positions[1:, 0:1] - global_positions[:-1, 0:1])
    root_rvelocity = Pivots.from_quaternions(root_rotation[1:] * -root_rotation[:-1]).ps
    root_velocity = root_velocity.repeat(njoints, axis=1)
    root_rvelocity = root_rvelocity.repeat(njoints, axis=1)[..., np.newaxis]

    # motion clip info
    positions = positions[:-1]
    rotations = rotations[:-1]
    root_trajectory = np.concatenate([root_velocity, root_rvelocity], axis=-1)
    motion_clip = np.concatenate([positions, rotations, root_trajectory], axis=-1) #here the motion format, global rotation relative to root
    # F, J, 3+4+4
    # feet contact info """
    motion_feet = feet_contact_from_positions(positions)

    return motion_clip, motion_feet


def generate_data(filename, downsample=1):
    dataframe, feet_cnt = preprocess(filename, slice=False, downsample=downsample)
    dataframe = np.transpose(dataframe, (2, 0, 1))  # (C, F, J)
    return dataframe, feet_cnt


def generate_dataset(data_dir, out_path, downsample=2, window=64, window_step=16):
    style_files = get_bvh_files(data_dir)
    
    style_clips = []
    style_feet = []
    style_classes = []
    for i, item in enumerate(style_files):
        print('Processing %i of %i (%s)' % (i, len(style_files), item))
        clip, feet, cls = preprocess(item, downsample=downsample, window=window, window_step=window_step)
        style_clips += clip  
        style_feet += feet 
        style_classes += cls

    style_clips = np.array(style_clips)
    style_feet = np.array(style_feet)
    style_clips = np.transpose(style_clips, (0, 3, 1, 2))
    np.savez_compressed(out_path, clips=style_clips, feet=style_feet, classes=style_classes)


def generate_mean_std(dataset_path, out_path):
    X = np.load(dataset_path)['clips']
        
    print('Total shape: ', X.shape)  # (N, C, F, J)
    X = X[:, :-4, :, :]  # (N, 7, F, J)
    Xmean = X.mean(axis=(0, 2), keepdims=True)[0]
    Xmean = np.concatenate([Xmean, np.zeros((4,) + Xmean.shape[1:])])
    Xstd = X.std(axis=(0, 2), keepdims=True)[0]
    idx = Xstd < 1e-5
    Xstd[idx] = 1
    Xstd = np.concatenate([Xstd, np.ones((4,) + Xstd.shape[1:])])

    print('Mean shape', Xmean.shape)
    print('Std shape: ', Xstd.shape)
    np.savez_compressed(out_path, Xmean=Xmean, Xstd=Xstd)


"""
if __name__ == '__main__':
    generate_dataset('../bvh/generate', '../datasets/styletransfer_generate', downsample=2, window=64, window_step=32)
    generate_mean_std('../datasets/styletransfer_generate.npz', '../datasets/preprocess_styletransfer_generate')

    generate_dataset('../bvh/classify', '../datasets/styletransfer_classify', downsample=2, window=64, window_step=32)
    generate_mean_std('../datasets/styletransfer_classify.npz', '../datasets/preprocess_styletransfer_classify')

    print('done!')
"""
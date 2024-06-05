import os
import sys
import numpy as np
import torch
import argparse
from tqdm import tqdm

from os.path import join as pjoin


from motion import BVH
from motion.InverseKinematics import JacobianInverseKinematics, BasicInverseKinematics
from scripts.motion_process_bvh import *
from motion.Animation import *


def softmax(x, **kw):
    softness = kw.pop("softness", 1.0)
    maxi, mini = np.max(x, **kw), np.min(x, **kw)
    return maxi + np.log(softness + np.exp(mini - maxi))


def softmin(x, **kw):
    return -softmax(-x, **kw)


def alpha(t):
    return 2.0 * t * t * t - 3.0 * t * t + 1


def lerp(a, l, r):
    return (1 - a) * l + a * r


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="bla_3d")
    return parser.parse_args()


def remove_fs(anim, glb, foot_contact, bvh_writer, output_path, fid_l=(3, 4), fid_r=(7, 8), interp_length=5, force_on_floor=True):
    raw_glb = glb.copy()
    scale = np.mean(raw_glb[:, 12]) / 1.65
    if foot_contact is None:
        def foot_detect(positions, thres):
            velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

            feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
            feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
            feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
            feet_l_h = positions[:-1, fid_l, 1]
            feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)

            feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
            feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
            feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
            feet_r_h = positions[:-1, fid_r, 1]

            feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)

            return feet_l, feet_r

        feet_thre = 0.05 * scale
        # feet_thre = 0.05
        feet_l, feet_r = foot_detect(glb, thres=feet_thre)
        foot = np.concatenate([feet_l, feet_r], axis=-1).transpose(1, 0)  # [4, T-1]
        foot = np.concatenate([foot, foot[:, -1:]], axis=-1)
    else:
        foot = foot_contact.transpose(1, 0)

    T = len(glb)

    fid = list(fid_l) + list(fid_r)
    fid_l, fid_r = np.array(fid_l), np.array(fid_r)
    foot_heights = np.minimum(glb[:, fid_l, 1],
                              glb[:, fid_r, 1]).min(axis=1)  # [T, 2] -> [T]

    floor_height = softmin(foot_heights, softness=0.5, axis=0)

    glb[:, :, 1] -= floor_height
    anim.positions[:, 0, 1] -= floor_height
    for i, fidx in enumerate(fid):
        fixed = foot[i]  # [T]

        """
        for t in range(T):
            glb[t, fidx][1] = max(glb[t, fidx][1], 0.25)
        """

        s = 0
        while s < T:
            while s < T and fixed[s] == 0:
                s += 1
            if s >= T:
                break
            t = s
            avg = glb[t, fidx].copy()
            while t + 1 < T and fixed[t + 1] == 1:
                t += 1
                avg += glb[t, fidx].copy()
            avg /= (t - s + 1)

            if force_on_floor:
                avg[1] = 0.0

            for j in range(s, t + 1):
                glb[j, fidx] = avg.copy()

            s = t + 1

        for s in range(T):
            if fixed[s] == 1:
                continue
            l, r = None, None
            consl, consr = False, False
            for k in range(interp_length):
                if s - k - 1 < 0:
                    break
                if fixed[s - k - 1]:
                    l = s - k - 1
                    consl = True
                    break
            for k in range(interp_length):
                if s + k + 1 >= T:
                    break
                if fixed[s + k + 1]:
                    r = s + k + 1
                    consr = True
                    break

            if not consl and not consr:
                continue
            if consl and consr:
                litp = lerp(alpha(1.0 * (s - l + 1) / (interp_length + 1)),
                            glb[s, fidx], glb[l, fidx])
                ritp = lerp(alpha(1.0 * (r - s + 1) / (interp_length + 1)),
                            glb[s, fidx], glb[r, fidx])
                itp = lerp(alpha(1.0 * (s - l + 1) / (r - l + 1)),
                           ritp, litp)
                glb[s, fidx] = itp.copy()
                continue
            if consl:
                litp = lerp(alpha(1.0 * (s - l + 1) / (interp_length + 1)),
                            glb[s, fidx], glb[l, fidx])
                glb[s, fidx] = litp.copy()
                continue
            if consr:
                ritp = lerp(alpha(1.0 * (r - s + 1) / (interp_length + 1)),
                            glb[s, fidx], glb[r, fidx])
                glb[s, fidx] = ritp.copy()

    targetmap = {}
    for j in range(glb.shape[1]):
        targetmap[j] = glb[:, j]

    ik = BasicInverseKinematics(anim, glb, iterations=3,
                                silent=True)

    ik()
    bvh_writer.write(output_path, np.array(anim.rotations), anim.positions[:, 0], order="zyx", names=anim.names)
    glb_fixed = positions_global(anim)
    return anim, glb_fixed


def compute_foot_sliding(foot_data, traj_qpos, offseth):
    foot = np.array(foot_data).copy()
    offseth = np.mean(foot[:10, 1])
    foot[:, 1] -= offseth  # Grounding it
    foot_disp = np.linalg.norm(foot[1:, [0, 2]] - foot[:-1, [0, 2]], axis=1)
    traj_qpos[:, 1] -= offseth
    seq_len = len(traj_qpos)
    H = 0.05
    y_threshold = 0.65  # yup system
    y = traj_qpos[1:, 1]

    foot_avg = (foot[:-1, 1] + foot[1:, 1]) / 2
    subset = np.logical_and(foot_avg < H, y > y_threshold)
    # import pdb; pdb.set_trace()

    sliding_stats = np.abs(foot_disp * (2 - 2 ** (foot_avg / H)))[subset]
    sliding = np.sum(sliding_stats) / seq_len * 1000
    return sliding, sliding_stats







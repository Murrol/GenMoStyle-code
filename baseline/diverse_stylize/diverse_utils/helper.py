import os
import sys
import numpy as np

sys.path.append('../')
from animation import BVH
from animation.InverseKinematics import JacobianInverseKinematics
from animation.Quaternions import Quaternions


###############################################################################
# Helper Functions
###############################################################################
def make_dir(parent_dir, dir_name):
    child_dir = os.path.join(parent_dir, dir_name)
    if not os.path.exists(child_dir):
        os.makedirs(child_dir)
    return child_dir


def make_dir_replicate(parent_dir, dir_name):
    file_id = 0
    for f in os.listdir(parent_dir):
        if f.startswith(dir_name):
            file_id += 1

    if file_id > 0:
        dir_name += '_(%d)' % file_id

    child_dir = make_dir(parent_dir, dir_name)
    return child_dir


###############################################################################
# Normalization & Denormalization
###############################################################################
def normalize(x, mean, std):
    x = (x - mean) / std
    return x


def denormalize(x, mean, std):
    x = x * std + mean
    return x


###############################################################################
# Smoothing
###############################################################################
def softmax(x, **kw):
    softness = kw.pop('softness', 1.0)
    maxi, mini = np.max(x, **kw), np.min(x, **kw)
    return maxi + np.log(softness + np.exp(mini - maxi))


def softmin(x, **kw):
    return -softmax(-x, **kw)


def alpha(t):
    return 2.0 * t * t * t - 3.0 * t * t + 1


def lerp(a, l, r):
    return (1 - a) * l + a * r


###############################################################################
# Restoring Animation & Saving into BVH
###############################################################################
def restore_animation(pos, traj, start=None, end=None):
    """
    :param pos: (F, J, 3)
    :param traj: (F, J, 4)
    :param start: start frame index
    :param end: end frame index
    :return: positions
    """
    if start is None:
        start = 0
    if end is None:
        end = len(pos)

    Rx = traj[start:end, 0, -4]
    Ry = traj[start:end, 0, -3]
    Rz = traj[start:end, 0, -2]
    Rr = traj[start:end, 0, -1]

    rotation = Quaternions.id(1)
    translation = np.array([[0, 0, 0]])

    for fi in range(len(pos)):
        pos[fi, :, :] = rotation * pos[fi]
        pos[fi] = pos[fi] + translation[0]  # NOTE: xyz-translation
        rotation = Quaternions.from_angle_axis(-Rr[fi], np.array([0, 1, 0])) * rotation
        translation = translation + rotation * np.array([Rx[fi], Ry[fi], Rz[fi]])
    global_positions = pos

    return global_positions


def to_bvh_cmu(targets, filename, silent=True, frametime=1.0/60.0):
    """
    from 21 to 31 joints
    """
    rest, names, _ = BVH.load('rest_cmu.bvh')
    anim = rest.copy()
    anim.positions = anim.positions.repeat(len(targets), axis=0)
    anim.rotations.qs = anim.rotations.qs.repeat(len(targets), axis=0)

    sdr_l, sdr_r, hip_l, hip_r = 13, 17, 1, 5
    across1 = targets[:, hip_l] - targets[:, hip_r]
    across0 = targets[:, sdr_l] - targets[:, sdr_r]
    across = across0 + across1
    across = across / np.sqrt((across ** 2).sum(axis=-1))[...,np.newaxis]

    forward = np.cross(across, np.array([[0,1,0]]))
    forward = forward / np.sqrt((forward**2).sum(axis=-1))[...,np.newaxis]
    target = np.array([[0,0,1]]).repeat(len(forward), axis=0)

    anim.positions[:,0] = targets[:,0]
    anim.rotations[:,0:1] = -Quaternions.between(forward, target)[:,np.newaxis]

    mapping = {
        0: 0,
        2: 1, 3: 2, 4: 3, 5: 4,
        7: 5, 8: 6, 9: 7, 10: 8,
        12: 9, 13: 10, 15: 11, 16: 12,
        18: 13, 19: 14, 20: 15, 22: 16,
        25: 17, 26: 18, 27: 19, 29: 20,
    }

    targetmap = {}
    for k in mapping:
        targetmap[k] = targets[:, mapping[k]]

    ik = JacobianInverseKinematics(anim, targetmap, iterations=10, damping=2.0, silent=silent)
    ik()

    BVH.save(filename, anim, names, frametime=frametime)


def remove_fs(glb, feet, fid_l=(3, 4), fid_r=(7, 8), interp_length=15, force_on_floor=False):
    T = len(glb)
    feet = np.transpose(feet, (1, 0))

    fid = list(fid_l) + list(fid_r)
    fid_l, fid_r = np.array(fid_l), np.array(fid_r)
    foot_heights = np.minimum(glb[:, fid_l, 1],
                              glb[:, fid_r, 1]).min(axis=1)
    floor_height = softmin(foot_heights, softness=0.5, axis=0)
    glb[:, :, 1] -= floor_height

    for i, fidx in enumerate(fid):
        fixed = feet[i]
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

            # print(fixed[s - 1:t + 2])

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

    return glb


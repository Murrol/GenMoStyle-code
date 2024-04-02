import os

# os.path.a
from tqdm import tqdm
from common.quaternion import *
import motion.BVH as BVH
from os.path import join as pjoin

def mirror_motion(source, target):
    animation = BVH.load(source)
    rotations = animation.rotations
    positions = animation.positions
    trajectories = positions[:, 0].copy()
    offsets = animation.offsets
    parents = animation.parents
    names = animation.names
    frametime = animation.frametime

    joints_left = np.array([1, 2, 3, 4, 5, 17, 18, 19, 20, 21, 22, 23], dtype='int64')
    joints_right = np.array([6, 7, 8, 9, 10, 24, 25, 26, 27, 28, 29, 30], dtype='int64')

    mirrored_rotations = rotations.copy()
    mirrored_trajectory = trajectories.copy()

    mirrored_rotations[:, joints_left] = rotations[:, joints_right]
    mirrored_rotations[:, joints_right] = rotations[:, joints_left]
    mirrored_rotations[:, :, [2, 3]] *= -1
    mirrored_rotations = qfix_np(mirrored_rotations)
    mirrored_rotations = qeuler_np(mirrored_rotations, order="zyx")

    mirrored_trajectory[:, 0] *= -1
    BVH.write_bvh(parents, offsets, mirrored_rotations,
                  mirrored_trajectory, names, frametime,
                  path=target, order="zyx")

if __name__ == "__main__":
    source_root = "../../data/motion_transfer/mocap_cmu"
    target_root = source_root
    fnames = os.listdir(source_root)
    # print(fnames)
    for name in tqdm(fnames):
        target_path = pjoin(target_root, "m_"+name)
        mirror_motion(os.path.join(source_root, name), target_path)
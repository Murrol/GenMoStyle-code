import numpy
import torch

from common.quaternion import *


class Skeleton(object):
    def __init__(self, offsets, parents, device):
        self.device = device
        if isinstance(offsets, numpy.ndarray):
            # (joint_nums, 3)
            self.offsets = torch.from_numpy(offsets).to(device).float()
        self.parents = parents
        self.children = [[] for _ in range(len(parents))]
        for i in range(len(self.parents)):
            if self.parents[i] >= 0:
                self.children[self.parents[i]].append(i)

    def get_forward_direction(self, across1, across2):
        across = across1 + across2
        across = across / np.sqrt((across ** 2).sum(axis=-1))[:, np.newaxis]
        # print(across1.shape, across2.shape)
        # forward (batch_size, 3)
        forward = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
        forward = forward / np.sqrt((forward ** 2).sum(axis=-1))[..., np.newaxis]
        return forward

    # Be sure root joint is at the beginning of kinematic chains
    def fk_local_quat(self, local_quats, root_pos):
        # local_quats (batch_size, joints_num, 4)
        # joints (batch_size, joints_num, 3)
        # root_pos (batch_size, 3)
        global_pos = torch.zeros(local_quats.shape[:-1] + (3,)).to(self.device)
        global_pos[:, 0] = root_pos
        global_quats = torch.zeros_like(local_quats).to(self.device)
        global_quats[:, 0] = local_quats[:, 0]
        offsets = self.offsets.expand(local_quats.shape[0], -1, -1).float()

        for i in range(1, len(self.parents)):
#             if self.offsets[i].sum() == 0:
#                 global_quats[:, i] = global_quats[:, self.parents[i]]
#                 global_pos[:, i] = global_pos[:, self.parents[i]]
#             else:
            global_quats[:, i] = qmul(global_quats[:, self.parents[i]], local_quats[:, i])
            global_pos[:, i] = qrot(global_quats[:, self.parents[i]], offsets[:, i]) + global_pos[:, self.parents[i]]
        return global_quats, global_pos

    def fk_local_quat_np(self, local_quat, root_pos):
        global_quats, global_pos =  self.fk_local_quat(torch.from_numpy(local_quat).float(),
                                                        torch.from_numpy(root_pos).float())
        return global_quats.numpy(), global_pos.numpy()

    def fk_global_quat(self, global_quats, root_pos):
        global_pos = torch.zeros(global_quats.shape[:-1] + (3,)).to(self.device)
        global_pos[:, 0] = root_pos
        offsets = self.offsets.expand(global_quats.shape[0], -1, -1).float()

        for i in range(1, len(self.parents)):
            global_pos[:, i] = qrot(global_quats[:, self.parents[i]], offsets[:, i]) + global_pos[:, self.parents[i]]
        return global_pos

    def fk_global_quat_np(self, global_quats, root_pos):
        global_pos = self.fk_global_quat(torch.from_numpy(global_quats).float(),
                                         torch.from_numpy(root_pos).float())
        return global_pos.numpy()


    def fk_local_cont6d(self, local_cont6d, root_pos):
        # cont6d_params (batch_size, joints_num, 6)
        # joints (batch_size, joints_num, 3)
        # root_pos (batch_size, 3)
        global_pos = torch.zeros(local_cont6d.shape[:-1]+(3,)).to(self.device)
        global_pos[:, 0] = root_pos

        local_cont6d_mat = cont6d_to_mat(local_cont6d)
        global_cont6d_mat = torch.zeros_like(local_cont6d_mat).to(self.device)
        global_cont6d_mat[:, 0] = local_cont6d_mat[:, 0]
        offsets = self.offsets.expand(local_cont6d.shape[0], -1, -1).float()


        for i in range(1, len(self.parents)):
#             if self.offsets[i].sum() == 0:
#                 global_cont6d_mat[:, i] = global_cont6d_mat[:, self.parents[i]]
#                 global_pos[:, i] = global_pos[:, self.parents[i]]
#             else:
            global_cont6d_mat[:, i] = torch.matmul(global_cont6d_mat[:, self.parents[i]],
                                                   local_cont6d_mat[:, i])
#             print(global_cont6d_mat[:, i].shape, offsets[:, i].unsqueeze(-1).shape)
            global_pos[:, i] = torch.matmul(global_cont6d_mat[:, self.parents[i]],
                                                offsets[:, i].unsqueeze(-1)).squeeze() + global_pos[:, self.parents[i]]
        return mat_to_cont6d(global_cont6d_mat), global_pos

    def fk_local_cont6d_np(self, local_cont6d, root_pos):
        global_cont6d, global_pos = self.fk_local_cont6d(torch.from_numpy(local_cont6d).float(),
                                                         torch.from_numpy(root_pos).float())
        return global_cont6d.numpy(), global_pos.numpy()

    def fk_global_cont6d(self, global_cont6d, root_pos):
        # cont6d_params (batch_size, joints_num, 6)
        # joints (batch_size, joints_num, 3)
        # root_pos (batch_size, 3)

        global_cont6d_mat = cont6d_to_mat(global_cont6d)
        global_pos = torch.zeros(global_cont6d.shape[:-1] + (3,)).to(self.device)
        global_pos[:, 0] = root_pos
        offsets = self.offsets.expand(global_cont6d.shape[0], -1, -1).float()

        for i in range(1, len(self.parents)):
            global_pos[:, i] = torch.matmul(global_cont6d_mat[:, self.parents[i]],
                                            offsets[:, i].unsqueeze(-1)).squeeze() + global_pos[:, self.parents[i]]
        return global_pos

    def fk_global_cont6d_np(self, global_cont6d, root_pos):
        global_pos = self.fk_global_cont6d(torch.from_numpy(global_cont6d).float(),
                                          torch.from_numpy(root_pos).float())
        return global_pos.numpy()

    # @classmethod
    def global_to_local_quat(self, global_quat):
        local_quat = torch.zeros_like(global_quat).to(global_quat.device)
        local_quat[:, 0] = global_quat[:, 0]

        for i in range(1, len(self.parents)):
            local_quat[:, i] = qmul(qinv(global_quat[:, self.parents[i]]), global_quat[:, i])
            # global_quats[:, i] = qmul(global_quats[:, self.parents[i]], local_quats[:, i])
        return local_quat

    def global_to_local_quat_np(self, global_quat):
        local_quat = self.global_to_local_quat(torch.from_numpy(global_quat).float())
        return local_quat.numpy()
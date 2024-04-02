import os
import sys

from options.test_options import TestOptions
from data import create_data_loader
from data.data_loader_gen import TestInputFetcher, SourceDataset, to
from model import create_model
from processor import Processor
from diverse_utils.helper import denormalize

sys.path.insert(0, os.getcwd())
BASEPATH = "../"
sys.path.insert(0, BASEPATH)
# print(BASEPATH, os.getcwd())
from os.path import join as pjoin
import common.paramUtil as paramUtil
import networks.networks as Net
from scripts.motion_process_bvh import *
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict
from utils.metrics import calculate_activation_statistics, calculate_frechet_distance, euclidean_distance_matrix, geodesic_distance

import shutil
from motion.Quaternions import Quaternions
from utils.plot_script import *
from utils.remove_fs import remove_fs
from datetime import datetime
import time

def create_CLS(opt):
    e_mid_channels = [dim_pose - 4, 512, 768]
    e_st_channels = [768, 512, 512]
    # num_clas
    classifier = Net.ResNetClassifier(e_mid_channels, e_st_channels, opt.num_of_style)
    classifier.to(opt.device)
    checkpoint = torch.load(pjoin(opt.checkpoints_dir, 'bfa', "CLS_FT1_ML160", "model", "best.tar"),
                            map_location=opt.device)
    classifier.load_state_dict(checkpoint["classifier"])
    classifier.eval()

    return classifier

if __name__ == "__main__":
    test_options = TestOptions()
    opt = test_options.parse()
    print('Start test on cuda:%s' % opt.gpu_ids)

    opt.use_ik = True
    # opt.use_ik = False
    datasetname = "bfa"
    saving = True
    time_buf = []

    # Load experiment setting
    
    opt.checkpoints_dir = '../../evaluation_files'
    bfa_data_root = "../../motion_transfer_data/processed_bfa"
    cmu_data_root = "../../motion_transfer_data/processed_cmu"

    # create dataset
    opt.batch_size = 1
    opt.dataroot = bfa_data_root
    ref_loader = SourceDataset(opt, type='test')
    if datasetname == "cmu":
        opt.dataroot = cmu_data_root
    src_loader = SourceDataset(opt, type='test')

    opt.use_action = False
    opt.use_style = True
    opt.num_of_action = 1
    opt.num_of_style = len(bfa_style_inv_enumerator)
    anim = BVH.load(pjoin(bfa_data_root, "bvh", "Hurried_02.bvh"))
    skeleton = Skeleton(anim.offsets, anim.parents, "cpu")
    style_enumerator = bfa_style_enumerator

    # opt.topology = paramUtil.parents
    action_dim = opt.num_of_action if opt.use_action else 0
    style_dim = opt.num_of_style if opt.use_style else 0

    now = datetime.now()
    
    opt.result_dir = pjoin(opt.save_dir, datasetname, opt.name+opt.alter+'_'+now.strftime('_%m%d_%H%M%S'))
    # opt.result_dir = pjoin(opt.save_dir, 'bfa', opt.name+'_'+now.strftime('_%m%d_%H%M%S'))
    opt.joint_dir = pjoin(opt.result_dir, 'joints')
    opt.animation_dir = pjoin(opt.result_dir, 'animations')
    os.makedirs(opt.joint_dir, exist_ok=True)
    os.makedirs(opt.animation_dir, exist_ok=True)

    bvh_writer = BVH.WriterWrapper(anim.parents, anim.frametime, anim.offsets, anim.names)

    # opt.use_skeleton = True
    opt.joint_num = 21
    kinematic_chain = kinematic_chain.copy()
    # opt.joint_num = len(kinematic_chain)
    radius = 40
    fps = 30
    dim_pose = 260

    mean = np.load(pjoin("../../motion_transfer_data/processed_bfa", "Mean.npy"))
    std = np.load(pjoin("../../motion_transfer_data/processed_bfa", "Std.npy"))

    # create model, trainer
    model = create_model(opt)
    tester = Processor(opt)

    if opt.load_latest:
        model.load_networks()
        opt.load_iter = model.get_current_iter()
    else:
        model.load_networks(opt.load_iter)
    print('Parameters/Optimizers are loaded from the iteration %d' % opt.load_iter)
    
    # Load experiment setting
    opt.device = torch.device("cpu" if eval(opt.gpu_ids)==-1 else f"cuda:{opt.gpu_ids}")
    torch.autograd.set_detect_anomaly(True)
    if eval(opt.gpu_ids) != -1:
        torch.cuda.set_device(eval(opt.gpu_ids))

    style_cls = create_CLS(opt)
    Xmean = tester.Xmean
    Xstd = tester.Xstd

    if os.path.exists(f'../random_selected_uids_{datasetname}.npz'):
        random_selected_uids = np.load(f'../random_selected_uids_{datasetname}.npz', allow_pickle=True)['data']
        gen_data = []
        for case_name in random_selected_uids:
            temp_name = case_name.replace("_CNT_", "")
            gen_data.append({"cnt": temp_name.split("_STY_")[0], "sty": temp_name.split("_STY_")[1]})
    else:
        gen_data = np.load(f'../gen_dataset_{datasetname}.npz', allow_pickle=True)['data']
    # gen_data = np.load('../gen_dataset_bfa.npz', allow_pickle=True)['data']

    def deskeletonize(motion, root_data, fc):
        joint_num = opt.joint_num
        # motion = motion.permute(0, 3, 2, 1)
        B, T, J, C = motion.shape
        # motion = motion.reshape(shape[:-1] + (joint_num, -1))
        
        positions = motion[..., :3].reshape([B, T, J*3])
        rotations = motion[..., 3:9].reshape([B, T, J*6])
        # velocities = motion[..., 9:12].reshape([B, T, J*3])
        # root_data = root_data.permute(0, 2, 1)
        foot_contact = fc
        #     print(positions.shape)
        # print(root_data.shape, positions.shape, rotations.shape, velocities.shape, foot_contact.shape)
        # print(root_data.shape, positions.shape, rotations.shape, foot_contact.shape)
        data = torch.concat([root_data, positions, rotations, foot_contact], dim=-1)
        return data

    t = 0
    device = torch.device("cuda:{}".format(opt.gpu_ids[0]) if torch.cuda.is_available() else "cpu")
    clip_idx = 0
    for gen_pair in gen_data:
        cnt_uid = gen_pair['cnt']
        sty_uid = gen_pair['sty']
        content_data = src_loader.indexing(gen_pair['cnt'], clip_idx)
        style_data = ref_loader.indexing(gen_pair['sty'], clip_idx)
        inputs = {}
        src = content_data
        
        inputs_src = {'x_real': src['x'], 'y_org': src['y'], 'c_real': src['c']}
        inputs.update(inputs_src)
        ref = style_data

        z = torch.randn(opt.latent_dim)   # random Gaussian noise for x_ref
        inputs_ref = {'x_ref': ref['x'],
                        'c_ref': ref['c'],
                        'y_trg': ref['y'],
                        'z_trg': z}
        inputs.update(inputs_ref)
        inputs = to(inputs, device, expand_dim=True)
        with torch.no_grad():
            torch.cuda.synchronize()
            start_time = time.time()
            output_ref = tester.test(model, inputs, alter=opt.alter)
            torch.cuda.synchronize()
            end_time = time.time()
            time_buf.append(end_time-start_time)
        fc = inputs['x_real']['feet']
        glbr = inputs['x_real']['traj']
        glbr2 = inputs['x_ref']['traj']
        fc2 = inputs['x_ref']['feet']

        SID1 = inputs['y_org'].detach().cpu().numpy()
        SID2 = inputs['y_trg'].detach().cpu().numpy()
        M1 = inputs['x_real']['posrot']
        M2 = inputs['x_ref']['posrot']
        
        TM = denormalize(output_ref, Xmean[:, :9], Xstd[:, :9])
        TM = torch.permute(TM, (0, 2, 3, 1))
        M1 = denormalize(M1, Xmean[:, :9], Xstd[:, :9])
        M1 = torch.permute(M1, (0, 2, 3, 1))
        M2 = denormalize(M2, Xmean[:, :9], Xstd[:, :9])
        M2 = torch.permute(M2, (0, 2, 3, 1))

        glbr = denormalize(glbr, Xmean[:, -4:], Xstd[:, -4:])
        glbr = torch.permute(glbr, (0, 2, 3, 1))[:, :, 0] # (C, F, J) -> (F, C)

        glbr2 = denormalize(glbr2, Xmean[:, -4:], Xstd[:, -4:])
        glbr2 = torch.permute(glbr2, (0, 2, 3, 1))[:, :, 0]

        M1 = deskeletonize(M1, glbr, fc).detach().cpu().numpy()
        M2 = deskeletonize(M2, glbr2, fc).detach().cpu().numpy()
        TM = deskeletonize(TM, glbr, fc).detach().cpu().numpy()
        tmvel = TM[:, 1:, 4: 4 + opt.joint_num * 3] - TM[:, :-1, 4: 4 + opt.joint_num * 3]
        tmvel = np.concatenate((tmvel, tmvel[:, [-1]]), axis=1)
        TM = np.concatenate((TM[:, :, :4 + opt.joint_num * 9], tmvel, TM[:, :, 4 + opt.joint_num * 9:]), axis=-1)
        TM = (TM - mean) / std

        tmvel = M1[:, 1:, 4: 4 + opt.joint_num * 3] - M1[:, :-1, 4: 4 + opt.joint_num * 3]
        tmvel = np.concatenate((tmvel, tmvel[:, [-1]]), axis=1)
        M1 = np.concatenate((M1[:, :, :4 + opt.joint_num * 9], tmvel, M1[:, :, 4 + opt.joint_num * 9:]), axis=-1)
        M1 = (M1 - mean) / std

        tmvel = M2[:, 1:, 4: 4 + opt.joint_num * 3] - M2[:, :-1, 4: 4 + opt.joint_num * 3]
        tmvel = np.concatenate((tmvel, tmvel[:, [-1]]), axis=1)
        M2 = np.concatenate((M2[:, :, :4 + opt.joint_num * 9], tmvel, M2[:, :, 4 + opt.joint_num * 9:]), axis=-1)
        M2 = (M2 - mean) / std

        _M2 = torch.from_numpy(M2).permute(0, 2, 1).float().to(opt.device)
        _TM = torch.from_numpy(TM).permute(0, 2, 1).float().to(opt.device)
        style_feat_GT, gt_pred = style_cls(_M2[:, :-4])
        style_feat_FK, fake_pred = style_cls(_TM[:, :-4])
        print(gt_pred.argmax(), fake_pred.argmax(), SID2)

        if not saving:
            continue
        B, L, D = M1.shape
        NM1 = M1 * std + mean
        NM2 = M2 * std + mean
        NTM = TM * std + mean
        
        b = 0
        case_name = f"_CNT_{cnt_uid}_STY_{sty_uid}"
        print(case_name)
        os.makedirs(pjoin(opt.animation_dir, case_name), exist_ok=True)
        os.makedirs(pjoin(opt.joint_dir, case_name), exist_ok=True)
        Style1 = style_enumerator[SID1[b]]
        Style2 = style_enumerator[SID2[b]]
        StyleN = style_enumerator[SID2[b]]

        m1 = recover_pos_from_rot(torch.from_numpy(NM1[b]).float(),
                                    opt.joint_num, skeleton).numpy()
        m2 = recover_pos_from_rot(torch.from_numpy(NM2[b]).float(),
                                    opt.joint_num, skeleton).numpy()
        tm = recover_pos_from_rot(torch.from_numpy(NTM[b]).float(),
                                    opt.joint_num, skeleton).numpy()

        _, lq_m1, rp_m1 = recover_bvh_from_rot(torch.from_numpy(NM1[b]).float(),
                                                opt.joint_num, skeleton)
        _, lq_m2, rp_m2 = recover_bvh_from_rot(torch.from_numpy(NM2[b]).float(),
                                    opt.joint_num, skeleton)
        _, lq_tm, rp_tm = recover_bvh_from_rot(torch.from_numpy(NTM[b]).float(),
                                    opt.joint_num, skeleton)

        if opt.use_ik:
            anim.rotations = Quaternions(lq_tm.numpy())
            positions = anim.positions[:len(rp_tm)]
            positions[:, 0] = rp_tm
            anim.positions = positions
            foot = np.zeros_like(NTM[b, :, -4:])
            foot[NTM[b, :, -4:] > 0.2] = 1
            foot[NTM[b, :, -4:] <= 0.2] = 0
            bvh, glb = remove_fs(anim, tm, foot, bvh_writer,
                                pjoin(opt.animation_dir, case_name, "TM_%s_%d.bvh" % (StyleN, t)))
            np.save(pjoin(opt.joint_dir, case_name, "TM_%s_%d.npy" % (StyleN, t)), NTM[b])
            # plot_3d_motion(pjoin(opt.animation_dir, case_name, "TM_%s_%d.mp4" % (StyleN, t)),
            #             kinematic_chain, glb, title=StyleN, fps=fps, radius=radius)
        else:
            np.save(pjoin(opt.joint_dir, case_name, "TM_%s_%d.npy"%(StyleN, t)), NTM[b])
            # plot_3d_motion(pjoin(opt.animation_dir, case_name, "TM_%s_%d.mp4" % (StyleN, t)),
            #             kinematic_chain, tm, title=StyleN, fps=fps, radius=radius)
            bvh_writer.write(pjoin(opt.animation_dir, case_name, "TM_%s_%d.bvh" % (StyleN, t)),
                            lq_tm.numpy(), rp_tm.numpy(), order="zyx")

        np.save(pjoin(opt.joint_dir, case_name, "M1_%s_%d.npy" % (Style1, t)), NM1[b])
        # plot_3d_motion(pjoin(opt.animation_dir, case_name, "M1_%s_%d.mp4"%(Style1, t)),
        #             kinematic_chain, m1, title=Style1, fps=fps, radius=radius)
        bvh_writer.write(pjoin(opt.animation_dir, case_name, "M1_%s_%d.bvh" % (Style1, t)),
                        lq_m1.numpy(), rp_m1.numpy(), order="zyx")
        # break
    time_buf.sort()
    time_buf_filtered = time_buf[int(len(time_buf)*0.2):-int(len(time_buf)*0.2)]
    inference_time = np.mean(time_buf_filtered) * 1000
    std = np.std(time_buf_filtered) * 1000
    print(f"Inference time for diverse_{opt.alter}:{inference_time:.4f}ms, std:{std:.4f}")
import os
import sys 
sys.path.insert(0, os.getcwd())
BASEPATH = "../"
sys.path.insert(0, BASEPATH)
# print(BASEPATH, os.getcwd())
from os.path import join as pjoin
import common.paramUtil as paramUtil
import networks.networks as Net
from trainer import Trainer
# from data.dataset import MotionDataset, MotionEvalDataset
from scripts.motion_process_bvh import *
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict
from utils.metrics import calculate_activation_statistics, calculate_frechet_distance, euclidean_distance_matrix, geodesic_distance
import argparse
import importlib
from data_loader_gen import MotionDataset
from motion import Animation
from etc.utils import set_seed, ensure_dirs, get_config
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
    checkpoint = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, "CLS_FT1_ML160", "model", "best.tar"),
                            map_location=opt.device)
    classifier.load_state_dict(checkpoint["classifier"])
    classifier.eval()

    return classifier, None, None

def initialize_path(args, config, save=True):
    config['main_dir'] = os.path.join('../../puzzle_exp', config['name'])
    config['model_dir'] = os.path.join(config['main_dir'], "pth")
    config['tb_dir'] = os.path.join(config['main_dir'], "log")
    config['info_dir'] = os.path.join(config['main_dir'], "info")
    config['output_dir'] = os.path.join(config['main_dir'], "output")
    ensure_dirs([config['main_dir'], config['model_dir'], config['tb_dir'],
                 config['info_dir'], config['output_dir']])
    if save:
        shutil.copy(args.config, os.path.join(config['info_dir'], 'config.yaml'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_eval.yaml',
                        help='Path to the config file.')
    args = parser.parse_args()

    """ initialize """
    optdict = get_config(args.config)
    initialize_path(args, optdict, save=False)

    for k, v in optdict.items():
        parser.add_argument('--'+k, default=v)
    opt = parser.parse_args()
    opt.use_ik = True
    dataset_name = "bfa"
    saving = True
    time_buf = []

    # Load experiment setting
    opt.device = torch.device("cpu" if opt.gpu_id==-1 else "cuda:%d"%(opt.gpu_id) )
    torch.autograd.set_detect_anomaly(True)
    if opt.gpu_id != -1:
        torch.cuda.set_device(opt.gpu_id)
    
    opt.checkpoints_dir = '../../evaluation_files'
    bfa_data_root = "../../motion_transfer_data/processed_bfa"
    cmu_data_root = "../../motion_transfer_data/processed_cmu"
    
    opt.use_action = False
    opt.num_of_action = 1
    opt.num_of_style = len(bfa_style_inv_enumerator)
    anim = BVH.load(pjoin(bfa_data_root, "bvh", "Hurried_02.bvh"))
    skeleton = Skeleton(anim.offsets, anim.parents, "cpu")
    style_enumerator = bfa_style_enumerator

    # opt.topology = paramUtil.parents
    action_dim = opt.num_of_action if opt.use_action else 0
    style_dim = opt.num_of_style if opt.use_style else 0

    now = datetime.now()
    opt.result_dir = pjoin(opt.main_dir, dataset_name, opt.name+'_'+now.strftime('_%m%d_%H%M%S'))
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
    # test_data_path = pjoin(opt.data_root, "test_data.npy")
    # trainer = SkeletonTrainer(opt, encoder, decoder)

    
    # test_dataset = MotionDataset(opt, mean, std, test_data_path, fix_bias=True)
    # # test_dataset.set_style(style_inv_enumerator["Heavy"], style_inv_enumerator["Old"])
    # data_loader = DataLoader(test_dataset, batch_size=opt.batch_size, num_workers=4,
    #                         drop_last=False, shuffle=True, pin_memory=True)

    optdict["data_dir"] = bfa_data_root
    style_data_loader = MotionDataset('test', optdict)
    if dataset_name == "cmu":
        optdict["data_dir"] = cmu_data_root
    content_data_loader = MotionDataset('test', optdict)
    if os.path.exists(f'../random_selected_uids_{dataset_name}.npz'):
        random_selected_uids = np.load(f'../random_selected_uids_{dataset_name}.npz', allow_pickle=True)['data']
        gen_data = []
        for case_name in random_selected_uids:
            temp_name = case_name.replace("_CNT_", "")
            gen_data.append({"cnt": temp_name.split("_STY_")[0], "sty": temp_name.split("_STY_")[1]})
    else:
        gen_data = np.load(f'../gen_dataset_{dataset_name}.npz', allow_pickle=True)['data']

    # Trainer
    trainer = Trainer(optdict)
    print("here!")
    trainer.to(opt.device)
    trainer.load_checkpoint()

    def deskeletonize(motion, root_data, fc):
        joint_num = opt.joint_num
        motion = motion.permute(0, 3, 2, 1)
        B, T, J, C = motion.shape
        # motion = motion.reshape(shape[:-1] + (joint_num, -1))
        
        positions = motion[..., :3].reshape([B, T, J*3])
        rotations = motion[..., 3:9].reshape([B, T, J*6])
        velocities = motion[..., 9:12].reshape([B, T, J*3])
        root_data = root_data.permute(0, 2, 1)
        foot_contact = fc
        #     print(positions.shape)
        # print(root_data.shape, positions.shape, rotations.shape, velocities.shape, foot_contact.shape)
        data = torch.concat([root_data, positions, rotations, velocities, foot_contact], dim=-1)
        return data

    t = 0
    clip_idx = 0
    for gen_pair in gen_data:
        content_data = content_data_loader.indexing(gen_pair['cnt'], clip_idx)
        style_data = style_data_loader.indexing(gen_pair['sty'], clip_idx)

        cnt_data, glbr, fc, cnt_uid = content_data["motion"][None], content_data["root"][None], content_data["foot_contact"][None], content_data["uid"]
        sty_data, sty_label, glbr2, sty_uid = style_data["motion"][None], style_data["label"], style_data["root"][None], style_data["uid"]
        with torch.no_grad():
            torch.cuda.synchronize()
            start_time = time.time()
            outputs = trainer.eval_test(cnt_data, sty_data)
            torch.cuda.synchronize()
            end_time = time.time()
            time_buf.append(end_time-start_time)
        # rec = outputs["recon_con"].squeeze()
        tra = outputs["stylized"]
        con_gt = outputs["con_gt"]
        sty_gt = outputs["sty_gt"]
        SID2 = [sty_label]
        SID1 = [content_data["label"]]
        M1 = con_gt
        M2 = sty_gt
        TM = tra
        M1 = deskeletonize(M1, glbr, fc).detach().cpu().numpy()
        M2 = deskeletonize(M2, glbr2, fc).detach().cpu().numpy()
        TM = deskeletonize(TM, glbr, fc).detach().cpu().numpy()
        
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
    print(f"Inference time for puzzle:{inference_time:.4f}ms, std:{std:.4f}")
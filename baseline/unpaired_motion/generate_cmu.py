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
from data_loader_gen import MotionNorm
from scripts.motion_process_bvh import *
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict
from utils.metrics import calculate_activation_statistics, calculate_frechet_distance, euclidean_distance_matrix, geodesic_distance
import argparse
import importlib
from animation_data import AnimationData
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--config', type=str, default='config_eval')
    parser.add_argument('--gpu_id', type=int, default=-1)
    parser.add_argument('--dataset_name', type=str, default='bfa', help='Dataset Name')
    parser.add_argument('--repeat_times', type=int, default=3, help="Number of generation rounds for each text description")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config_module = importlib.import_module(args.config)
    opt = config_module.Config()

    opt.use_ik = True
    # opt.use_ik = False
    datasetname = "bfa"
    saving = True
    time_buf = []

    # Load experiment setting
    opt.initialize(args)
    opt.device = torch.device("cpu" if opt.gpu_id==-1 else "cuda:%d"%(opt.gpu_id) )
    torch.autograd.set_detect_anomaly(True)
    if opt.gpu_id != -1:
        torch.cuda.set_device(opt.gpu_id)

    opt.checkpoints_dir = '../../evaluation_files'
    bfa_data_root = "../../motion_transfer_data/processed_bfa"
    cmu_data_root = "../../motion_transfer_data/processed_cmu"

    # create dataset
    opt.batch_size = 1
    opt.data_dir = bfa_data_root
    style_data_loader = MotionNorm(opt, 'test')
    if datasetname == "cmu":
        opt.data_dir = cmu_data_root
    content_data_loader = MotionNorm(opt, 'test')

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
    
    opt.result_dir = pjoin(opt.main_dir, datasetname, opt.name+'_'+now.strftime('_%m%d_%H%M%S'))
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

    # Trainer
    trainer = Trainer(opt)
    print("here!")
    trainer.to(opt.device)
    trainer.resume()

    style_cls = create_CLS(opt)

    if os.path.exists(f'../random_selected_uids_{datasetname}.npz'):
        random_selected_uids = np.load(f'../random_selected_uids_{datasetname}.npz', allow_pickle=True)['data']
        gen_data = []
        for case_name in random_selected_uids:
            temp_name = case_name.replace("_CNT_", "")
            gen_data.append({"cnt": temp_name.split("_STY_")[0], "sty": temp_name.split("_STY_")[1]})
    else:
        gen_data = np.load(f'../gen_dataset_{datasetname}.npz', allow_pickle=True)['data']
    # gen_data = np.load('../gen_dataset_bfa.npz', allow_pickle=True)['data']

    def to(inputs, device, expand_dim=False):
        for name, ele in inputs.items():
            if isinstance(ele, dict):
                for k, v in ele.items():
                    if expand_dim:
                        v = torch.unsqueeze(torch.tensor(v), dim=0)
                    ele[k] = v.to(device, dtype=torch.float)
            else:
                if expand_dim:
                    ele = torch.unsqueeze(torch.tensor(ele), dim=0)
                if name.startswith('label') or name.startswith('action'):
                    inputs[name] = ele.to(device, dtype=torch.long)
                else:
                    inputs[name] = ele.to(device, dtype=torch.float)
        return inputs

    t = 0
    clip_idx = 0
    for gen_pair in gen_data:
        cnt_uid = gen_pair['cnt']
        sty_uid = gen_pair['sty']
        content_data = content_data_loader.indexing(gen_pair['cnt'], clip_idx)
        style_data = style_data_loader.indexing(gen_pair['sty'], clip_idx)
        data = {
                "style3draw": content_data["style3draw"],
                "contentraw": content_data["contentraw"],
                "content": content_data["content"],
                "style3d": content_data["style3d"],
                "label": content_data["label"],
                "foot_contact": content_data["foot_contact"],

                "diff_style3d": style_data["style3d"],
                "label_diff": style_data["label"],
                "diff_style3d_nrot": style_data["contentraw"]
            }
        data = to(data, opt.device, expand_dim=True)
        with torch.no_grad():
            torch.cuda.synchronize()
            start_time = time.time()
            re_dict = trainer.eval_test(data)
            torch.cuda.synchronize()
            end_time = time.time()
            time_buf.append(end_time-start_time)
        # print(data["contentraw"].shape, re_dict["trans"].shape)
        SID1 = data["label"].detach().cpu().numpy()
        SID2 = data["label_diff"].detach().cpu().numpy()
        
        TM = re_dict["trans"].detach().cpu().numpy()
        # print(TM.shape, data["contentraw"].shape)
        _anim = AnimationData.from_network_output(TM.squeeze())
        _anim, names, ftime = _anim.get_BVH()
        TM = (process_data_full(_anim, 1, [5, 1, 17, 13], 0.05) - mean) / std
        TM = np.concatenate([TM, TM[[-1]]], axis=0)
        M1 = data["contentraw"].detach().cpu().numpy()
        _anim = AnimationData.from_network_output(M1.squeeze())
        _anim, names, ftime = _anim.get_BVH()
        M1 = (process_data_full(_anim, 1, [5, 1, 17, 13], 0.05) - mean) / std
        M1 = np.concatenate([M1, M1[[-1]]], axis=0)
        M2 = data["diff_style3d_nrot"].detach().cpu().numpy()
        _anim = AnimationData.from_network_output(M2.squeeze())
        _anim, names, ftime = _anim.get_BVH()
        M2 = (process_data_full(_anim, 1, [5, 1, 17, 13], 0.05) - mean) / std
        M2 = np.concatenate([M2, M2[[-1]]], axis=0)

        TM, M1, M2 = TM[None], M1[None], M2[None]

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
    print(f"Inference time for unpair:{inference_time:.4f}ms, std:{std:.4f}")
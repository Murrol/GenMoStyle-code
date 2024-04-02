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
from data_loader_my import get_dataloader
from animation_data import AnimationData
from motion import Animation
os.environ["OMP_NUM_THREADS"] = "1"

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

    # style_matcher = Net.ContrastiveFeatureExtractor(e_mid_channels, e_st_channels)
    # style_matcher.to(opt.device)
    # checkpoint = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, "Style_Matcher", "model", "best.tar"),
    #                         map_location=opt.device)
    # style_matcher.load_state_dict(checkpoint["model"])
    # style_matcher.eval()
    #
    # content_matcher = Net.ContrastiveFeatureExtractor(e_mid_channels, e_st_channels)
    # content_matcher.to(opt.device)
    # checkpoint = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, "Content_Matcher", "model", "best.tar"),
    #                         map_location=opt.device)
    # content_matcher.load_state_dict(checkpoint["model"])
    # content_matcher.eval()
    return classifier, None, None

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

    # Load experiment setting
    opt.initialize(args)
    opt.device = torch.device("cpu" if opt.gpu_id==-1 else "cuda:%d"%(opt.gpu_id) )
    torch.autograd.set_detect_anomaly(True)
    if opt.gpu_id != -1:
        torch.cuda.set_device(opt.gpu_id)
    
    opt.checkpoints_dir = '../../evaluation_files'
    # opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    # opt.model_dir = pjoin(opt.save_root, 'model')
    # opt.meta_dir = pjoin(opt.save_root, 'meta')

    # opt.result_dir = pjoin(opt.result_path, opt.dataset_name, opt.name, opt.ext)
    # opt.joint_dir = pjoin(opt.result_dir, 'joints')
    # opt.animation_dir = pjoin(opt.result_dir, 'animations')
    # # os.makedirs(opt.joint_dir, exist_ok=True)
    # # os.makedirs(opt.animation_dir, exist_ok=True)

    if opt.dataset_name == 'bfa':
        opt.data_root = "../../motion_transfer_data/processed_bfa"
        opt.use_action = False
        opt.num_of_action = 1
        style_enumerator = bfa_style_enumerator
        style_inv_enumerator = bfa_style_inv_enumerator
        opt.num_of_style = len(bfa_style_inv_enumerator)
        anim = BVH.load(pjoin(opt.data_root, "bvh", "Hurried_02.bvh"))
        skeleton = Skeleton(anim.offsets, anim.parents, "cpu")
        # joint_num =
        # opt.motion_length = 96
    elif opt.dataset_name == "xia":
        opt.data_root = "../../motion_transfer_data/processed_xia/"
        opt.num_of_action = len(xia_action_inv_enumerator)
        opt.num_of_style = len(xia_style_inv_enumerator)
        style_enumerator = xia_style_enumerator
        style_inv_enumerator = xia_style_inv_enumerator
        anim = BVH.load(pjoin(opt.data_root, "bvh", "angry_transitions_001.bvh"))
        skeleton = Skeleton(anim.offsets, anim.parents, "cpu")
    else:
        raise Exception("Unsupported data type !~")

    opt.topology = paramUtil.parents
    action_dim = opt.num_of_action if opt.use_action else 0
    style_dim = opt.num_of_style if opt.use_style else 0

    # opt.use_skeleton = True
    opt.joint_num = 21
    kinematic_chain = kinematic_chain.copy()
    # opt.joint_num = len(kinematic_chain)
    radius = 40
    fps = 30
    dim_pose = 260

    classifier, style_matcher, content_matcher = create_CLS(opt)


    mean = np.load(pjoin("../../motion_transfer_data/processed_bfa", "Mean.npy"))
    std = np.load(pjoin("../../motion_transfer_data/processed_bfa", "Std.npy"))
    # test_data_path = pjoin(opt.data_root, "test_data.npy")
    # trainer = SkeletonTrainer(opt, encoder, decoder)

    
    # test_dataset = MotionDataset(opt, mean, std, test_data_path, fix_bias=True)
    # # test_dataset.set_style(style_inv_enumerator["Heavy"], style_inv_enumerator["Old"])
    # data_loader = DataLoader(test_dataset, batch_size=opt.batch_size, num_workers=4,
    #                         drop_last=False, shuffle=True, pin_memory=True)

    data_loader = get_dataloader(opt, 'test', shuffle=True)

    # Trainer
    trainer = Trainer(opt)
    print("here!")
    trainer.to(opt.device)
    trainer.resume()


    def get_metric_statistics(values):
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0)
        conf_interval = 1.96 * std / np.sqrt(opt.repeat_times)
        return mean, conf_interval

    res = OrderedDict({"S_FID":[], "G_DIS":[],"GT_ACC":[], "FK_ACC":[]})
    # res = OrderedDict(defaultdict(list))
    for t in range(opt.repeat_times):
        s_gt_feats = []
        s_fake_feats = []
        # c_gt_feats = []
        # c_fake_feats = []
        geo_dists = []
        gt_preds = []
        fake_preds = []
        gt_labels = []
        for i, data in tqdm(enumerate(data_loader)):
            
            # M1, _, _, M2, A1, S1, SID1, S2, SID2 = data
            # M2, A2, S2, SID2 = trainer.swap(M1), trainer.swap(A1), trainer.swap(S1), trainer.swap(SID1)
            # TM = trainer.generate(M1, M2, A1, A1, S1, S2, opt.sampling, opt.label_switch)
            re_dict = trainer.eval_test(data)
            SID2 = data["label_diff"]
            # SID2 = data["label"]
            TM = re_dict["trans"].detach().cpu().numpy()
            # print(TM.shape, data["contentraw"].shape)
            anim = AnimationData.from_network_output(TM.squeeze())
            anim, names, ftime = anim.get_BVH()
            TM = (process_data_full(anim, 1, [5, 1, 17, 13], 0.05) - mean) / std
            M1 = data["contentraw"].detach().cpu().numpy()
            anim = AnimationData.from_network_output(M1.squeeze())
            anim, names, ftime = anim.get_BVH()
            M1 = (process_data_full(anim, 1, [5, 1, 17, 13], 0.05) - mean) / std
            M2 = data["diff_style3d_nrot"].detach().cpu().numpy()
            # M2 = data["contentraw"].detach().cpu().numpy()
            anim = AnimationData.from_network_output(M2.squeeze())
            anim, names, ftime = anim.get_BVH()
            M2 = (process_data_full(anim, 1, [5, 1, 17, 13], 0.05) - mean) / std

            TM, M1, M2 = TM[None], M1[None], M2[None]
            B, L, D = M1.shape
            source_motion = TM * std + mean
            target_motion = M1 * std + mean
            source_rot6d = source_motion[..., 4 + opt.joint_num * 3: 4 + opt.joint_num * 9].reshape(B, L, -1, 6)
            target_rot6d = target_motion[..., 4 + opt.joint_num * 3: 4 + opt.joint_num * 9].reshape(B, L, -1, 6)
            source_rotmat = cont6d_to_mat(torch.from_numpy(source_rot6d))
            target_rotmat = cont6d_to_mat(torch.from_numpy(target_rot6d))
            geo_dist = geodesic_distance(source_rotmat, target_rotmat, reduction="none").mean([1, 2])
            # geo_dist = geo_dist
            geo_dists.append(geo_dist)

            M2 = torch.from_numpy(M2).permute(0, 2, 1).float().to(opt.device)
            TM = torch.from_numpy(TM).permute(0, 2, 1).float().to(opt.device)
            # M1 = M1.permute(0, 2, 1).float().to(opt.device)
            style_feat_GT, gt_pred = classifier(M2[:, :-4])
            style_feat_FK, fake_pred = classifier(TM[:, :-4])
            # content_feat_GT, _ = classifier(M1[:, :-4])
            # content_feat_FK = style_feat_FK.clone()

            # style_feat_GT = style_matcher(M2[:, :-4])
            # style_feat_FK = style_matcher(TM[:, :-4])
            #
            # content_feat_GT = content_matcher(M1[:, :-4])
            # content_feat_FK = content_matcher(TM[:, :-4])

            s_gt_feats.append(style_feat_GT)
            s_fake_feats.append(style_feat_FK)


            # c_gt_feats.append(content_feat_GT)
            # c_fake_feats.append(content_feat_FK)

            gt_preds.append(gt_pred)
            fake_preds.append(fake_pred)
            gt_labels.append(SID2)
        s_gt_feats = torch.cat(s_gt_feats, dim=0).detach().cpu().numpy()
        s_fake_feats = torch.cat(s_fake_feats, dim=0).detach().cpu().numpy()

        # c_gt_feats = torch.cat(c_gt_feats, dim=0).detach().cpu().numpy()
        # c_fake_feats = torch.cat(c_fake_feats, dim=0).detach().cpu().numpy()


        gt_pred_labels = torch.cat(gt_preds, dim=0).argmax(dim=-1).detach().cpu().numpy()
        fake_pred_labels = torch.cat(fake_preds, dim=0).argmax(dim=-1).detach().cpu().numpy()
        gt_labels = torch.cat(gt_labels, dim=0).detach().cpu().numpy()

        g_dist = torch.cat(geo_dists, dim=0).mean().item()

        s_gt_mu, s_gt_cov = calculate_activation_statistics(s_gt_feats)
        s_fk_mu, s_fk_cov = calculate_activation_statistics(s_fake_feats)
        # print(gt_mu, fk_mu)
        s_fid = calculate_frechet_distance(s_gt_mu, s_gt_cov, s_fk_mu, s_fk_cov)
        # s_dis = euclidean_distance_matrix(s_gt_feats, s_fake_feats).mean()

        # c_gt_mu, c_gt_cov = calculate_activation_statistics(c_gt_feats)
        # c_fk_mu, c_fk_cov = calculate_activation_statistics(c_fake_feats)
        # print(gt_mu, fk_mu)
        # c_fid = calculate_frechet_distance(c_gt_mu, c_gt_cov, c_fk_mu, c_fk_cov)
        # c_dis = euclidean_distance_matrix(c_gt_feats, c_fake_feats).mean()

        # geo_dists
        # print(gt_pred_labels)
        # print(gt_labels)
        # print(fake_pred_labels)
        gt_accuracy = (gt_pred_labels == gt_labels).sum() / len(gt_labels)
        fk_accuracy = (fake_pred_labels == gt_labels).sum() / len(gt_labels)

        # print(s_fid, s_dis, c_fid, c_dis, gt_accuracy, fk_accuracy)
        # print("Time:%02d, S_FID:%.03f, S_DIS:%.03f, C_FID:%.03f, C_DIS:%.03f, GT_ACC:%.03f, FK_ACC:%.03f"%
        #       (t, s_fid, s_dis, c_fid, c_dis, gt_accuracy, fk_accuracy))
        print("Time:%02d, S_FID:%.03f, G_DIS:%.03f, GT_ACC:%.03f, FK_ACC:%.03f" %
              (t, s_fid, g_dist, gt_accuracy, fk_accuracy))
        res["S_FID"].append(s_fid)
        res["G_DIS"].append(g_dist)
        res["GT_ACC"].append(gt_accuracy)
        res["FK_ACC"].append(fk_accuracy)
    print(f"------------Summary motion unpair bfa {opt.motion_length}--------------")
    for key, value in res.items():
        mean, confInt = get_metric_statistics(value)
        print("%s, Mean:%.03f, Cint:%.03f"%(key, mean, confInt))
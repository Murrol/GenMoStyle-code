import os
import sys 

from options.test_options import TestOptions
from data import create_data_loader
from data.data_loader_my import TestInputFetcher
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
from collections import OrderedDict, defaultdict
from utils.metrics import calculate_activation_statistics, calculate_frechet_distance, euclidean_distance_matrix, geodesic_distance, calculate_multimodality
import argparse
import importlib

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

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--name', type=str)
#     parser.add_argument('--batch_size', type=int)
#     parser.add_argument('--config', type=str, default='config_eval')
#     parser.add_argument('--gpu_id', type=int, default=-1)
#     parser.add_argument('--dataset_name', type=str, default='bfa', help='Dataset Name')
#     parser.add_argument('--repeat_times', type=int, default=3, help="Number of generation rounds for each text description")
#     return parser.parse_args()

if __name__ == "__main__":
    test_options = TestOptions()
    opt = test_options.parse()
    print('Start test on cuda:%s' % opt.gpu_ids)

    # create dataset
    src_loader = create_data_loader(opt, which='source', type='test')
    ref_loader = create_data_loader(opt, which='reference', type='test')
    print('Training data loaded')
    dataset_size = len(src_loader)
    print('The number of training data = %d' % dataset_size)
    fetcher = TestInputFetcher(opt, src_loader, ref_loader)

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
    
    opt.checkpoints_dir = '../../evaluation_files'

    opt.dataset_name = 'bfa'
    opt.use_style = True
    opt.use_action = False
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
    Xmean = tester.Xmean
    Xstd = tester.Xstd

    # test_dataset = MotionDataset(opt, mean, std, test_data_path, fix_bias=True)
    # # test_dataset.set_style(style_inv_enumerator["Heavy"], style_inv_enumerator["Old"])
    # data_loader = DataLoader(test_dataset, batch_size=opt.batch_size, num_workers=4,
    #                         drop_last=False, shuffle=True, pin_memory=True)

    def get_metric_statistics(values):
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0)
        conf_interval = 1.96 * std / np.sqrt(opt.repeat_times)
        return mean, conf_interval

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

    res = OrderedDict({"S_FID":[], "G_DIS":[],"GT_ACC":[], "FK_ACC":[], "DIV":[]})
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
        mm_feats = []

        mm_sample_per = 10
        mm_num_repeats = 30
        mm_num_times = 10

        for i in tqdm(range(dataset_size)):
            inputs = next(fetcher)
            # stylize with a reference motion
            output_ref = tester.test(model, inputs, alter=opt.alter)
            # stylize with a random noise
            # output_latent = tester.test(model, inputs, alter='latent')
            fc = inputs['x_real']['feet']
            glbr = inputs['x_real']['traj']
            glbr2 = inputs['x_ref']['traj']
            fc2 = inputs['x_ref']['feet']

            SID1 = inputs['y_org']
            SID2 = inputs['y_trg']
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
            B, L, D = M1.shape

            tmvel = TM[:, 1:, 4: 4 + opt.joint_num * 3] - TM[:, :-1, 4: 4 + opt.joint_num * 3]
            tmvel = np.concatenate((tmvel, tmvel[:, [-1]]), axis=1)
            # print(TM.shape, tmvel.shape)
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

            if opt.sampling and i%mm_sample_per==0:
                inputs_M = fetcher.repeat(inputs, mm_num_repeats)
                # stylize with a reference motion
                output_ref_M = tester.test(model, inputs_M, alter=opt.alter)
                # stylize with a random noise
                # output_latent = tester.test(model, inputs, alter='latent')
                fc_M = inputs_M['x_real']['feet']
                glbr_M = inputs_M['x_real']['traj']

                TMM = denormalize(output_ref_M, Xmean[:, :9], Xstd[:, :9])
                TMM = torch.permute(TMM, (0, 2, 3, 1))

                glbr_M = denormalize(glbr_M, Xmean[:, -4:], Xstd[:, -4:])
                glbr_M = torch.permute(glbr_M, (0, 2, 3, 1))[:, :, 0] # (C, F, J) -> (F, C)

                # print(TMM.shape, glbr_M.shape, fc_M.shape)
                TMM = deskeletonize(TMM, glbr_M, fc_M).detach().cpu().numpy()
                B, L, D = TMM.shape

                tmvelM = TMM[:, 1:, 4: 4 + opt.joint_num * 3] - TMM[:, :-1, 4: 4 + opt.joint_num * 3]
                tmvelM = np.concatenate((tmvelM, tmvelM[:, [-1]]), axis=1)
                # print(TM.shape, tmvel.shape)
                TMM = np.concatenate((TMM[:, :, :4 + opt.joint_num * 9], tmvelM, TMM[:, :, 4 + opt.joint_num * 9:]), axis=-1)
                TMM = (TMM - mean) / std

                TMM = torch.from_numpy(TMM).float().to(opt.device).permute(0, 2, 1)
                # M1 = M1.permute(0, 2, 1).float().to(opt.device)
                style_feat_MM, _ = classifier(TMM[:, :-4])
                style_feat_MM = style_feat_MM.view(-1, mm_num_repeats, style_feat_MM.shape[-1])
                # MS2 = MS2.view(-1, mm_num_times, MS2.shape[-1])
                # print(MS2[0])
                mm_feats.append(style_feat_MM)

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

        if len(mm_feats)!=0:
            mm_feats = torch.cat(mm_feats, dim=0).detach().cpu().numpy()
            diversity = calculate_multimodality(mm_feats, mm_num_times)
        else:
            diversity = 0

        # print(s_fid, s_dis, c_fid, c_dis, gt_accuracy, fk_accuracy)
        # print("Time:%02d, S_FID:%.03f, S_DIS:%.03f, C_FID:%.03f, C_DIS:%.03f, GT_ACC:%.03f, FK_ACC:%.03f"%
        #       (t, s_fid, s_dis, c_fid, c_dis, gt_accuracy, fk_accuracy))
        print("Time:%02d, S_FID:%.03f, G_DIS:%.03f, GT_ACC:%.03f, FK_ACC:%.03f, DIV:%.03f" %
              (t, s_fid, g_dist, gt_accuracy, fk_accuracy, diversity))
        res["S_FID"].append(s_fid)
        res["G_DIS"].append(g_dist)
        res["GT_ACC"].append(gt_accuracy)
        res["FK_ACC"].append(fk_accuracy)
        res["DIV"].append(diversity)
    print(f"------------Summary motion diverse bfa {opt.motion_length} iter {opt.name}_{opt.load_iter}--------------")
    for key, value in res.items():
        mean, confInt = get_metric_statistics(value)
        print("%s, Mean:%.03f, Cint:%.03f"%(key, mean, confInt))
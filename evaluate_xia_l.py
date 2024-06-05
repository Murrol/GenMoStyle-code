import os
from os.path import join as pjoin
import common.paramUtil as paramUtil
from options.evaluate_vae_options import TestOptions
import networks.networks as Net
from networks.trainer import LatentVAETrainer
from data.dataset import MotionBfaXiaEvalDataset
from scripts.motion_process_bvh import *
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict
from utils.metrics import calculate_activation_statistics, calculate_frechet_distance, calculate_multimodality, geodesic_distance


def create_models(opt):
    en_channels = [dim_pose - 4, 384, 512]
    de_channels = [opt.dim_z, 512, 384]

    ae_encoder = Net.MotionEncoder(en_channels, opt.dim_z, vae_encoder=opt.use_vae)
    ae_decoder = Net.MotionDecoder(de_channels, output_size=dim_pose)
    encoder = Net.StyleContentEncoder(e_mid_channels, e_sp_channels, e_st_channels)
    generator = Net.Generator(n_conv, n_up, opt.dim_z, g_channels, dim_style)

    return ae_encoder, ae_decoder, encoder, generator

def create_GMR(opt):
    channels = [dim_pose - 4 - 2, 512, 256, 128, 64]
    regressor = Net.GlobalRegressor(3, 2, channels)
    regressor.to(opt.device)
    checkpoint = torch.load(pjoin(opt.checkpoints_dir, "cmu", "GLR_CV3_NP5_NS5_FT1", "model", "best.tar"),
                            map_location=opt.device)
    regressor.load_state_dict(checkpoint["regressor"])
    regressor.eval()
    return regressor

def create_CLS(opt):
    e_mid_channels = [dim_pose - 4, 512, 768]
    e_st_channels = [768, 512, 512]
    # num_clas
    classifier = Net.ResNetClassifier(e_mid_channels, e_st_channels, opt.num_of_style)
    classifier.to(opt.device)
    checkpoint = torch.load(pjoin(opt.checkpoints_dir, "bfa", "CLS_FT1_ML160", "model", "best.tar"),
                            map_location=opt.device)
    classifier.load_state_dict(checkpoint["classifier"])
    classifier.eval()

    action_recognizer = Net.GRUClassifier(dim_pose-4, opt.num_of_action, 512)
    action_recognizer.to(opt.device)
    checkpoint = torch.load(pjoin(opt.checkpoints_dir, "xia", "ACT_CLS_FT1", "model", "best.tar"),
                            map_location=opt.device)
    action_recognizer.load_state_dict(checkpoint["classifier"])
    action_recognizer.eval()

    return classifier, action_recognizer

if __name__ == "__main__":
    parser = TestOptions()
    opt = parser.parse()

    opt.device = torch.device("cpu" if opt.gpu_id==-1 else "cuda:%d"%(opt.gpu_id) )
    torch.autograd.set_detect_anomaly(True)
    if opt.gpu_id != -1:
        torch.cuda.set_device(opt.gpu_id)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')

    bfa_data_root = "../data/motion_transfer/processed_bfa"
    xia_data_root = "../data/motion_transfer/processed_xia"

    opt.use_action = False
    opt.num_of_action = len(xia_action_inv_enumerator)
    opt.num_of_style = len(bfa_style_inv_enumerator)
    anim = BVH.load(pjoin(bfa_data_root, "bvh", "Hurried_02.bvh"))
    skeleton = Skeleton(anim.offsets, anim.parents, "cpu")

    action_dim = opt.num_of_action if opt.use_action else 0
    style_dim = opt.num_of_style if opt.use_style else 0

    opt.joint_num = 21
    kinematic_chain = kinematic_chain.copy()

    radius = 40
    fps = 30
    dim_pose = 260

    target_channel = dim_pose
    # Encoder
    # 96 -> 48 -> 24
    e_mid_channels = [opt.dim_z, 768]
    e_sp_channels = [768 + action_dim, 512]
    e_st_channels = [768 + style_dim, 768, 512]

    dim_style = e_st_channels[-1] + style_dim
    # Generator
    n_conv = 2
    n_up = len(e_mid_channels) - 1
    g_channels = [e_sp_channels[-1] + action_dim, 768, 1024, 768, opt.dim_z]
    ae_encoder, ae_decoder, encoder, decoder = create_models(opt)
    gm_regressor = create_GMR(opt)
    style_cls, action_cls = create_CLS(opt)

    mean = np.load(pjoin(opt.meta_dir, "mean.npy"))
    std = np.load(pjoin(opt.meta_dir, "std.npy"))
    tc_mean = torch.from_numpy(mean).float().to(opt.device)
    tc_std = torch.from_numpy(std).float().to(opt.device)

    mean_cls = np.load(pjoin(bfa_data_root, "Mean.npy"))
    std_cls = np.load(pjoin(bfa_data_root, "Std.npy"))
    tc_mean_cls = torch.from_numpy(mean_cls).float().to(opt.device)
    tc_std_cls = torch.from_numpy(std_cls).float().to(opt.device)


    def transform(M):
        raw_m = M * tc_std + tc_mean
        return (raw_m - tc_mean_cls) / tc_std_cls

    xia_data_path = pjoin(xia_data_root, "test_data.npy")
    bfa_data_path = pjoin(bfa_data_root, "test_data.npy")
    trainer = LatentVAETrainer(opt, encoder, decoder, ae_encoder, ae_decoder)

    opt.batch_size = 1

    test_dataset = MotionBfaXiaEvalDataset(opt, mean, std, xia_data_path, bfa_data_path)
    # test_dataset.set_style(style_inv_enumerator["Heavy"], style_inv_enumerator["Old"])
    data_loader = DataLoader(test_dataset, batch_size=opt.batch_size, num_workers=1,
                            drop_last=False, shuffle=True, pin_memory=True)

    trainer.resume(pjoin(opt.model_dir, opt.which_epoch+".tar"))
    trainer.to([encoder, decoder, ae_encoder, ae_decoder], device=opt.device)
    trainer.net_eval([encoder, decoder, ae_encoder, ae_decoder])


    def get_metric_statistics(values):
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0)
        conf_interval = 1.96 * std / np.sqrt(opt.repeat_times)
        return mean, conf_interval

    res = OrderedDict({"C_FID":[], "G_DIS":[],"GT_C_ACC":[], "FK_C_ACC":[], "FK_S_ACC":[], "DIV":[]})
    # res = OrderedDict(defaultdict(list))
    for t in range(opt.repeat_times):
        s_gt_feats = []
        s_fake_feats = []
        c_gt_feats = []
        c_fake_feats = []
        geo_dists = []
        # gt_s_preds = []
        fake_s_preds = []
        gt_c_preds = []
        fake_c_preds = []
        gt_s_labels = []
        gt_c_labels = []

        mm_feats = []

        mm_sample_per = 10
        mm_num_repeats = 30
        mm_num_times = 10
        for i, data in enumerate(data_loader):
            M1, M2, A1, AID1, S2, SID2 = data

            TM = trainer.generate(M1, M2, S2, sampling=opt.sampling)
            gm_input = torch.cat([M1[..., 0:1].float().to(opt.device), TM[..., 3:]], dim=-1)[..., :-4]
            gm_input = gm_input.permute(0, 2, 1)
            root_vel = gm_regressor(gm_input)
            root_vel = root_vel.permute(0, 2, 1)
            M1 = M1.float().to(opt.device)
            TM = torch.cat([M1[..., 0:1], root_vel, TM[..., 3:]], dim=-1)

            if opt.sampling and i%mm_sample_per==0:
                MM1 = M1[:, None].repeat(1, mm_num_repeats, 1, 1).view(-1, M1.shape[1], M1.shape[2])
                MM2 = M2[:, None].repeat(1, mm_num_repeats, 1, 1).view(-1, M2.shape[1], M2.shape[2])
                MS2 = S2[:, None].repeat(1, mm_num_repeats, 1).view(-1, S2.shape[1])

                TMM = trainer.generate(MM1, MM2, MS2, opt.sampling)
                gm_input = torch.cat([MM1[..., 0:1].float().to(opt.device), TMM[..., 3:]], dim=-1)[..., :-4]
                gm_input = gm_input.permute(0, 2, 1)
                root_vel = gm_regressor(gm_input)
                root_vel = root_vel.permute(0, 2, 1)
                MM1 = MM1.float().to(opt.device)
                TMM = torch.cat([MM1[..., 0:1], root_vel, TMM[..., 3:]], dim=-1)

                TMM = TMM.float().to(opt.device).permute(0, 2, 1)
                style_feat_MM, _ = style_cls(TMM[:, :-4])
                style_feat_MM = style_feat_MM.view(-1, mm_num_repeats, style_feat_MM.shape[-1])
                MS2 = MS2.view(-1, mm_num_times, MS2.shape[-1])

                mm_feats.append(style_feat_MM)

            B, L, D = M1.shape
            source_motion = test_dataset.inv_transform(TM.detach().cpu().numpy())
            target_motion = test_dataset.inv_transform(M1.detach().cpu().numpy())
            source_rot6d = source_motion[..., 4 + opt.joint_num * 3: 4 + opt.joint_num * 9].reshape(B, L, -1, 6)
            target_rot6d = target_motion[..., 4 + opt.joint_num * 3: 4 + opt.joint_num * 9].reshape(B, L, -1, 6)
            source_rotmat = cont6d_to_mat(torch.from_numpy(source_rot6d))
            target_rotmat = cont6d_to_mat(torch.from_numpy(target_rot6d))
            geo_dist = geodesic_distance(source_rotmat, target_rotmat, reduction="none").mean([1, 2])
            # geo_dist = geo_dist
            geo_dists.append(geo_dist)

            M1 = M1.float().to(opt.device)

            M1 = transform(M1)
            TM = transform(TM)
            M2 = transform(M2.to(opt.device))

            gt_c_feat, gt_c_pred = action_cls(M1[..., :-4])
            fake_c_feat, fake_c_pred = action_cls(TM[..., :-4])


            M2 = M2.permute(0, 2, 1).float().to(opt.device)
            TM = TM.permute(0, 2, 1).float()

            _, fake_s_pred = style_cls(TM[:, :-4])

            fake_s_preds.append(fake_s_pred)
            gt_s_labels.append(SID2)


            c_gt_feats.append(gt_c_feat)
            c_fake_feats.append(fake_c_feat)
            gt_c_preds.append(gt_c_pred)
            fake_c_preds.append(fake_c_pred)
            gt_c_labels.append(AID1)

        c_gt_feats = torch.cat(c_gt_feats, dim=0).detach().cpu().numpy()
        c_fake_feats = torch.cat(c_fake_feats, dim=0).detach().cpu().numpy()

        fake_s_pred_labels = torch.cat(fake_s_preds, dim=0).argmax(dim=-1).detach().cpu().numpy()
        gt_s_labels = torch.cat(gt_s_labels, dim=0).detach().cpu().numpy()

        fake_c_pred_labels = torch.cat(fake_c_preds, dim=0).argmax(dim=-1).detach().cpu().numpy()
        gt_c_pred_labels = torch.cat(gt_c_preds, dim=0).argmax(dim=-1).detach().cpu().numpy()
        gt_c_labels = torch.cat(gt_c_labels, dim=0).detach().cpu().numpy()

        g_dist = torch.cat(geo_dists, dim=0).mean().item()

        c_gt_mu, c_gt_cov = calculate_activation_statistics(c_gt_feats)
        c_fk_mu, c_fk_cov = calculate_activation_statistics(c_fake_feats)

        c_fid = calculate_frechet_distance(c_gt_mu, c_gt_cov, c_fk_mu, c_fk_cov)

        gt_c_accuracy = (gt_c_pred_labels == gt_c_labels).sum() / len(gt_c_labels)
        fk_c_accuracy = (fake_c_pred_labels == gt_c_labels).sum() / len(gt_c_labels)
        fk_s_accuracy = (fake_s_pred_labels == gt_s_labels).sum() / len(gt_s_labels)

        if len(mm_feats) != 0:
            mm_feats = torch.cat(mm_feats, dim=0).detach().cpu().numpy()
            diversity = calculate_multimodality(mm_feats, mm_num_times)
        else:
            diversity = 0

        print("Time:%02d, C_FID:%.03f, G_DIS:%.03f, GT_C_ACC:%.03f, FK_C_ACC:%.03f, FK_S_ACC:%.03f, DIV:%.03f" %
              (t, c_fid, g_dist, gt_c_accuracy, fk_c_accuracy, fk_s_accuracy, diversity))
        res["C_FID"].append(c_fid)
        res["G_DIS"].append(g_dist)
        res["GT_C_ACC"].append(gt_c_accuracy)
        res["FK_C_ACC"].append(fk_c_accuracy)
        res["FK_S_ACC"].append(fk_s_accuracy)
        res["DIV"].append(diversity)
    print("------------Summary--------------")
    for key, value in res.items():
        mean, confInt = get_metric_statistics(value)
        print("%s, Mean:%.03f, Cint:%.03f"%(key, mean, confInt))
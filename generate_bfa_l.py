import os
from os.path import join as pjoin
import common.paramUtil as paramUtil
from options.evaluate_vae_options import TestOptions

from utils.plot_script import *

import networks.networks as Net
from networks.trainer import LatentVAETrainer
from data.dataset import MotionDataset, MotionEvalDataset
from scripts.motion_process_bvh import *
from torch.utils.data import DataLoader
from utils.remove_fs import remove_fs
from motion.Quaternions import Quaternions

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

    opt.result_dir = pjoin(opt.result_path, opt.dataset_name, opt.name, opt.ext)
    opt.joint_dir = pjoin(opt.result_dir, 'joints')
    opt.animation_dir = pjoin(opt.result_dir, 'animations')
    os.makedirs(opt.joint_dir, exist_ok=True)
    os.makedirs(opt.animation_dir, exist_ok=True)

    if opt.dataset_name in ['bfa', 'cmu']:
        opt.data_root = "../data/motion_transfer/processed_bfa"
        opt.use_action = False
        opt.num_of_action = 1
        style_enumerator = bfa_style_enumerator
        style_inv_enumerator = bfa_style_inv_enumerator
        opt.num_of_style = len(bfa_style_inv_enumerator)
        anim = BVH.load(pjoin(opt.data_root, "bvh", "Hurried_02.bvh"))
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

    target_channel = dim_pose
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

    mean = np.load(pjoin(opt.meta_dir, "mean.npy"))
    std = np.load(pjoin(opt.meta_dir, "std.npy"))
    test_data_path = pjoin(opt.data_root, "test_data.npy")
    trainer = LatentVAETrainer(opt, encoder, decoder, ae_encoder, ae_decoder)

    bvh_writer = BVH.WriterWrapper(anim.parents, anim.frametime, anim.offsets, anim.names)

    # test_dataset = MotionEvalDataset(opt, mean, std, test_data_path)
    # test_dataset.set_content_id(opt.content_id)
    test_dataset = MotionDataset(opt, mean, std, test_data_path)
    # test_dataset.set_style(style_inv_enumerator["Depressed"], style_inv_enumerator["Old"])
    data_loader = DataLoader(test_dataset, batch_size=opt.batch_size, num_workers=4,
                            drop_last=True, shuffle=True, pin_memory=True)

    trainer.resume(pjoin(opt.model_dir, opt.which_epoch + ".tar"))
    trainer.to([encoder, decoder, ae_encoder, ae_decoder], device=opt.device)
    trainer.net_eval([encoder, decoder, ae_encoder, ae_decoder])

    "Generate Results"
    # result_dict = {}
    with torch.no_grad():
        # for it in range(opt.niter):
        for i, data in enumerate(data_loader):
            if i >= opt.niters:
                break
            if not opt.sampling:
                opt.repeat_times = 1
            M1, _, _, M2, A1, S1, SID1, S2, SID2 = data
            A2 = A1
            for t in range(opt.repeat_times):
                TM = trainer.generate(M1, M2, S2, opt.sampling)
                # Remove root velocity, foot contact
                gm_input = torch.cat([M1[..., 0:1].float().to(opt.device), TM[..., 3:]], dim=-1)[..., :-4]
                gm_input = gm_input.permute(0, 2, 1)
                root_vel = gm_regressor(gm_input)
                root_vel = root_vel.permute(0, 2, 1)
                TM = torch.cat([M1[..., 0:1].float().to(opt.device), root_vel, TM[..., 3:]], dim=-1)

                NM1 = test_dataset.inv_transform(M1.cpu().numpy())
                NM2 = test_dataset.inv_transform(M2.cpu().numpy())
                NTM = test_dataset.inv_transform(TM.cpu().numpy())

                # print(M1.shape, M2.shape, TM.shape)

                for b in range(opt.batch_size):
                    print("%02d_%02d_%02d"%(i, b, t))
                    os.makedirs(pjoin(opt.animation_dir, "%02d_%02d"%(i, b)), exist_ok=True)
                    os.makedirs(pjoin(opt.joint_dir, "%02d_%02d"%(i, b)), exist_ok=True)
                    Style1 = style_enumerator[SID1[b].item()]
                    Style2 = style_enumerator[SID2[b].item()]
                    StyleN = style_enumerator[SID2[b].item()]

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

                    if opt.sampling and (not opt.use_style):
                        StyleN = "Random"


                    if opt.use_ik:
                        anim.rotations = Quaternions(lq_tm.numpy())
                        positions = anim.positions[:len(rp_tm)]
                        positions[:, 0] = rp_tm
                        anim.positions = positions
                        foot = np.zeros_like(NTM[b, :, -4:])
                        foot[NTM[b, :, -4:] > 0.2] = 1
                        foot[NTM[b, :, -4:] <= 0.2] = 0
                        bvh, glb = remove_fs(anim, tm, foot, bvh_writer,
                                             pjoin(opt.animation_dir, "%02d_%02d" % (i, b), "TM_%s_%d.bvh" % (StyleN, t)))
                        np.save(pjoin(opt.joint_dir, "%02d_%02d" % (i, b), "TM_%s_%d.npy" % (StyleN, t)), NTM[b])
                        plot_3d_motion(pjoin(opt.animation_dir, "%02d_%02d" % (i, b), "TM_%s_%d.mp4" % (StyleN, t)),
                                       kinematic_chain, glb, title=StyleN, fps=fps, radius=radius)
                    else:
                        np.save(pjoin(opt.joint_dir, "%02d_%02d" % (i, b), "TM_%s_%d.npy"%(StyleN, t)), NTM[b])
                        plot_3d_motion(pjoin(opt.animation_dir, "%02d_%02d" % (i, b), "TM_%s_%d.mp4" % (StyleN, t)),
                                       kinematic_chain, tm, title=StyleN, fps=fps, radius=radius)
                        bvh_writer.write(pjoin(opt.animation_dir, "%02d_%02d" % (i, b), "TM_%s_%d.bvh" % (StyleN, t)),
                                         lq_tm.numpy(), rp_tm.numpy(), order="zyx")

                    if t == 0:
                        np.save(pjoin(opt.joint_dir, "%02d_%02d" % (i, b), "M1_%s_%d.npy" % (Style1, t)), NM1[b])
                        plot_3d_motion(pjoin(opt.animation_dir, "%02d_%02d" % (i, b), "M1_%s_%d.mp4"%(Style1, t)),
                                       kinematic_chain, m1, title=Style1, fps=fps, radius=radius)
                        bvh_writer.write(pjoin(opt.animation_dir, "%02d_%02d" % (i, b), "M1_%s_%d.bvh" % (Style1, t)),
                                         lq_m1.numpy(), rp_m1.numpy(), order="zyx")


                    if not opt.sampling:
                        plot_3d_motion(pjoin(opt.animation_dir, "%02d_%02d" % (i, b), "M2_%s_%d.mp4" % (Style2, t)),
                                       kinematic_chain, m2, title=Style2, fps=fps, radius=radius)
                        np.save(pjoin(opt.joint_dir, "%02d_%02d" % (i, b), "M2_%s_%d.npy" % (Style2, t)), NM2[b])
                        bvh_writer.write(pjoin(opt.animation_dir, "%02d_%02d" % (i, b), "M2_%s_%d.bvh" % (Style2, t)),
                                         lq_m2.numpy(), rp_m2.numpy(), order="zyx")
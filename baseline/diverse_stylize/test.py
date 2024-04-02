import os
import numpy as np
import torch

from options.test_options import TestOptions
from data.data_loader import TestInputFetcher
from model import create_model
from processor import Processor
from utils.logger import Logger


f = open('contents.txt', 'r')
contents = [line.strip() for line in f.readlines()]

f = open('styles.txt', 'r')
styles = [line.strip() for line in f.readlines()]

output_dir = 'output/'
src_file = os.path.join(output_dir, 'walking_neutral.bvh')
ref_file = os.path.join(output_dir, 'jumping_old.bvh')


if __name__ == '__main__':
    test_options = TestOptions()
    opt = test_options.parse()
    print('Start test on cuda:%s' % opt.gpu_ids)

    fetcher = TestInputFetcher(opt)

    # create model, trainer, logger
    model = create_model(opt)
    tester = Processor(opt)
    logger = Logger(opt)

    if opt.load_latest:
        model.load_networks()
        opt.load_iter = model.get_current_iter()
    else:
        model.load_networks(opt.load_iter)
    print('Parameters/Optimizers are loaded from the iteration %d' % opt.load_iter)

    cls_name = os.path.split(src_file)[1][:-4]
    src_con = contents.index(cls_name.split('_')[0])
    src_sty = styles.index(cls_name.split('_')[1])

    cls_name = os.path.split(ref_file)[1][:-4]
    ref_con = contents.index(cls_name.split('_')[0])
    ref_sty = styles.index(cls_name.split('_')[1])

    inputs = {}
    src_input = fetcher.get_data(src_file, sty=src_sty, con=src_con, type='src')
    ref_input = fetcher.get_data(ref_file, sty=ref_sty, con=ref_con, start=0, end=64, type='ref')
    latent_input = fetcher.get_latent()
    inputs.update(src_input)
    inputs.update(ref_input)
    inputs.update(latent_input)

    # stylize with a reference motion
    output_ref = tester.test(model, inputs, alter='ref')
    # stylize with a random noise
    output_latent = tester.test(model, inputs, alter='latent')

    output_ref_file = os.path.join(output_dir, 'output_ref.bvh')
    output_latent_file = os.path.join(output_dir, 'output_latent.bvh')
    logger.save_output(output_ref, inputs['x_real']['traj'], inputs['x_real']['feet'][0].cpu().numpy(), filename=output_ref_file, fs_fix=True)
    logger.save_output(output_latent, inputs['x_real']['traj'], inputs['x_real']['feet'][0].cpu().numpy(), filename=output_latent_file, fs_fix=True)
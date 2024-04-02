import time
from options.train_options import TrainOptions
from data import create_data_loader
from data.data_loader_my import InputFetcher
from model import create_model
from processor import Processor
from utils.logger import Logger
import os
os.environ["OMP_NUM_THREADS"] = "1"

if __name__ == '__main__':
    train_options = TrainOptions()
    opt = train_options.parse()
    print('Start training on cuda:%s' % opt.gpu_ids)

    # create dataset
    src_loader = create_data_loader(opt, which='source', type='train')
    ref_loader = create_data_loader(opt, which='reference', type='train')
    print('Training data loaded')
    dataset_size = len(src_loader)
    print('The number of training data = %d' % dataset_size)
    fetcher = InputFetcher(opt, src_loader, ref_loader)

    # create model, trainer, logger
    model = create_model(opt)
    trainer = Processor(opt)
    logger = Logger(opt)

    # load model when resume training
    if opt.load_latest:
        model.load_networks()
        opt.resume_iter = model.get_current_iter()
    elif opt.resume_iter > 0:
        model.load_networks(opt.resume_iter)

    # print networks
    if opt.net_print:
        model.print_networks()
    train_options.print_options(opt)

    # train!
    start_time = time.time()
    for iter in range(opt.resume_iter, opt.total_iters):
        inputs = next(fetcher)

        d_loss, d_losses_latent = trainer.compute_d_loss(model, inputs, alter='latent')
        model.reset_grad()
        d_loss.backward()
        model.optimizer_D.step()

        d_loss, d_losses_ref = trainer.compute_d_loss(model, inputs, alter='ref')
        model.reset_grad()
        d_loss.backward()
        model.optimizer_D.step()

        g_loss, g_losses_latent, output_latent = trainer.compute_g_loss(model, inputs, alter='latent')
        model.reset_grad()
        g_loss.backward()
        model.optimizer_G.step()
        model.optimizer_F.step()
        model.optimizer_E.step()

        g_loss, g_losses_ref, output_ref = trainer.compute_g_loss(model, inputs, alter='ref')
        model.reset_grad()
        g_loss.backward()
        model.optimizer_G.step()

        # print losses
        losses = {'d_losses_latent': d_losses_latent,
                  'd_losses_ref': d_losses_ref,
                  'g_losses_latent': g_losses_latent,
                  'g_losses_ref': g_losses_ref}

        if (iter + 1) % opt.print_every == 0:
            elapsed_time = time.time() - start_time
            message = '=====' * 20 + '\n'
            message += 'Elapsed time: %.3f, Iteration: [%d/%d]\n' % (elapsed_time, (iter + 1), opt.total_iters)
            message += logger.print_current_lrs(model.get_current_lrs())
            message += logger.print_current_weights()
            message += '-----' * 20 + '\n'
            message += logger.print_current_losses(iter + 1, losses)
            print(message)

        # save the model
        if (iter + 1) % opt.save_every == 0:
            model.save_networks(iter + 1)

        # save the latest model
        if (iter + 1) % opt.save_latest_every == 0:
            model.save_networks(iter + 1, latest=True)

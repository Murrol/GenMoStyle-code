import os
from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def __init__(self):
        BaseOptions.__init__(self)

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        # dataset parameter
        parser.add_argument('--num_workers', type=int, default=0)
        parser.add_argument('--dataroot', type=str, default="../../motion_transfer_data/processed_bfa", help='path to training set')
        parser.add_argument('--preproot', type=str, default='./datasets/preprocess_styletransfer_generate.npz', help='path to preprocess')
        # parser.add_argument('--clip_size', type=int, nargs='+', default=[64, 21])
        parser.add_argument('--motion_length', type=int, default=160)
        parser.add_argument('--clip_size', type=int, nargs='+', default=[160, 21])

        # training parameters
        parser.add_argument('--batch_size', type=int, default=8)
        parser.add_argument('--resume_iter', type=int, default=0)
        parser.add_argument('--total_iters', type=int, default=100000)
        parser.add_argument('--g_lr', type=float, default=1e-4, help='learning rate for G')
        parser.add_argument('--d_lr', type=float, default=1e-6, help='learning rate for D')
        parser.add_argument('--e_lr', type=float, default=1e-6, help='learning rate for E')
        parser.add_argument('--f_lr', type=float, default=1e-5, help='learning rate for F')
        parser.add_argument('--lr_decay_every', type=int, default=100, help='learning rate decay step size')
        parser.add_argument('--beta1', type=float, default=0.9)
        parser.add_argument('--beta2', type=float, default=0.99)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--lambda_adv', type=float, default=1.0)
        parser.add_argument('--lambda_reg', type=float, default=1.0, help='weight for R1 regularization')
        parser.add_argument('--lambda_con', type=float, default=1.0, help='weight for content reconstruction loss')
        parser.add_argument('--lambda_sty', type=float, default=1.0, help='weight for style reconstruction loss')
        parser.add_argument('--lambda_ds', type=float, default=1.0, help='weight for style diversification loss')
        parser.add_argument('--lambda_cyc', type=float, default=1.0, help='weight for cycle loss')
        parser.add_argument('--lambda_feet', type=float, default=1.0)
        parser.add_argument('--ds_iter', type=int, default=100000)

        # saving & loading
        parser.add_argument('--net_print', type=bool, default=True)
        parser.add_argument('--print_every', type=int, default=10)
        parser.add_argument('--save_every', type=int, default=5000)
        parser.add_argument('--save_latest_every', type=int, default=100)
        parser.add_argument('--load_latest', action='store_true')

        return parser

    def check(self, opt):
        assert opt.mode == 'train', 'Not a train mode!'
        assert opt.num_domains == len(opt.domains), 'Number of domains does not match!'

    def print_options(self, opt):
        message = BaseOptions.print_options(self, opt)
        file_name = os.path.join(opt.save_dir, '%s_opt.txt' % opt.mode)

        with open(file_name, 'a') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
            
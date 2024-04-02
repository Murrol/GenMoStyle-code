import argparse
from datetime import datetime
from diverse_utils.logger import make_dir

f = open('contents_xia.txt', 'r')
contents = [line.strip() for line in f.readlines()]

f = open('styles_bfa.txt', 'r')
styles = [line.strip() for line in f.readlines()]


class BaseOptions:
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # basic parameters
        parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
        parser.add_argument('--name', type=str, default='experiment_name')
        parser.add_argument('--checkpoints_dir', type=str, default='../../diverse_exp')
        parser.add_argument('--num_domains', type=int, default=len(styles))
        parser.add_argument('--domains', type=str, nargs='+', default=styles)

        # content domains for recongnition
        parser.add_argument('--num_contents', type=int, default=len(contents))
        parser.add_argument('--contents', type=str, nargs='+', default=contents)

        # model parameters
        parser.add_argument('--model', type=str, default='motion_graph_gan')
        # parser.add_argument('--input_nc', type=int, default=7)
        # parser.add_argument('--output_nc', type=int, default=7)
        parser.add_argument('--input_nc', type=int, default=9)
        parser.add_argument('--output_nc', type=int, default=9)
        parser.add_argument('--ngf', type=int, default=64)
        parser.add_argument('--nef', type=int, default=64)
        parser.add_argument('--ndf', type=int, default=64)
        parser.add_argument('--latent_dim', type=int, default=16)
        parser.add_argument('--hidden_dim', type=int, default=512)
        parser.add_argument('--style_dim', type=int, default=64)
        parser.add_argument('--ng_blk', type=int, default=2)
        parser.add_argument('--ng_btn', type=int, default=0)
        parser.add_argument('--nd_blk', type=int, default=2)
        parser.add_argument('--nd_btn', type=int, default=0)
        parser.add_argument('--ne_blk', type=int, default=2)
        parser.add_argument('--ne_btn', type=int, default=0)

        # misc
        parser.add_argument('--gpu_ids', type=str, default='0')
        parser.add_argument('--opt_print', type=bool, default=True)

        self.initialized = True
        return parser

    def gather_options(self):
        parser = None
        if not self.initialized:
            parser = argparse.ArgumentParser()
            parser = self.initialize(parser)
        self.parser = parser
        return parser.parse_args()

    def check(self, opt):
        pass

    def parse(self):
        opt = self.gather_options()
        opt.save_dir = make_dir(opt.checkpoints_dir, opt.name)
        self.check(opt)
        return opt

    def print_options(self, opt):
        now = datetime.now()
        message = ''
        message += '----------------- %s options -----------------\n' % (opt.mode).capitalize()
        message += '{}_start: {}\n'.format(opt.mode, now.strftime('%Y/%m/%d %H:%M:%S'))
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '[default: %s]' % str(default)
            message += '{}: {} {}\n'.format(str(k), str(v), comment)
        message += '----------------- End -----------------'

        if opt.opt_print:
            print(message)

        return message


from options.base_vae_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
        self.parser.add_argument('--repeat_times', type=int, default=3, help="Number of generation rounds for each text description")
        self.parser.add_argument('--which_epoch', type=str, default="best", help='Checkpoint that will be used')

        self.parser.add_argument('--result_path', type=str, default="./eval_results/", help='Path to save generation results')
        self.parser.add_argument('--niters', type=int, default=1, help='Number of descriptions that will be used')
        self.parser.add_argument('--ext', type=str, default='default', help='Save file path extension')

        self.parser.add_argument('--sampling', action="store_true", help='models are saved here')
        self.parser.add_argument('--use_ik', action="store_true", help='models are saved here')

        self.parser.add_argument('--label_switch', action="store_true", help='models are saved here')
        self.parser.add_argument('--content_id', type=int, default=65, help='Number of descriptions that will be used')



        self.is_train = False
from .options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--phase', type=str, default='train', help='phase to use (either test or train)')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--train_result_dir', type=str, default='./train_result', help='data directory')
        parser.add_argument('--num_epochs', type=int, default=100, help='the number of epochs to train the model')
        parser.add_argument('--start_epoch', type=int, default=0, help='the number of epochs to start training')
        parser.add_argument('--gan_mode', type=str, default='original', help='choose gan loss mode from [ls, original, w, hinge]')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for Adam')
        parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
        parser.add_argument('--lr_decay_iters', type=int, default=500, help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weigth of GAN loss')
        parser.add_argument('--lambda_content', type=float, default=0.05, help='weight of content loss')
        parser.add_argument('--lambda_style', type=float, default=500, help='weight of style loss')
        # Display and saving options
        parser.add_argument('--save_img_freq', type=int, default=4000, help='frequency of saving images')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        self.isTrain = True
        return parser

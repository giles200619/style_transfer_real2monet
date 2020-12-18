from .options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--phase', type=str, default='test', help='phase to use (either test or train)')
        parser.add_argument('--print_freq', type=int, default=10, help='frequency of showing progress')
        parser.add_argument('--results_dir', type=str, default='./results', help='saves results here')
        self.isTrain = False
        return parser

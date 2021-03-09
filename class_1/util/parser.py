import argparse


def getParser():
        
    #####
    ##  Main Arguments
    #####
    
    parser = argparse.ArgumentParser(description='PyTorch arguments')
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        help='default: resnet18')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--checkpoint-file', metavar='DIR', dest='checkpoint_file', 
                        help='path to model checkpoint')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for injection')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://1.1.1.1:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or '
                            'multi node data parallel training')
    parser.add_argument('-l', '--log', dest='log', action='store_true',
                        help='turn loging on')
    parser.add_argument('--log-path', dest='log_path', default=None, type=str,
                        help='path to log folder')
    parser.add_argument('--log-prefix', dest='log_prefix', default=None, type=str,
                        help='prefix of log output folder')
    
    #####
    ##  Training Arguments
    #####
    
    train_group = parser.add_argument_group('Arguments for model training')
    
    train_group.add_argument('--epochs', default=90, type=int, metavar='N',
                             help='number of total epochs to run')
    train_group.add_argument('-lr', '--learning-rate', dest='lr', default=0.1, type=float,
                             metavar='LR', help='initial learning rate')
    train_group.add_argument('-mm', '--momentum', dest='momentum', default=0.9, type=float, metavar='M',
                             help='momentum')
    train_group.add_argument('-gm', '--gamma', dest='gamma', default=0.1, type=float, metavar='M',
                             help='gamma')
    train_group.add_argument('-wd', '--weight-decay', dest='weight_decay', default=1e-4, type=float,
                             metavar='W', help='weight decay (default: 1e-4)')
    train_group.add_argument('--resume', default='', type=str, metavar='PATH',
                             help='path to latest checkpoint (default: none)')
    train_group.add_argument('-tb', '--test-batch-size', dest='test_batch_size', type=int, default=1000, metavar='N',
                             help='input batch size for testing (default: 1000)')
    train_group.add_argument('--save-model', dest='save_model', default=None, type=str,
                             help='save the current model with specified file name')


    return parser
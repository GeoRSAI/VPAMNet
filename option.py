#!/usr/bin/env Python
# coding=utf-8
import torchvision.models as models
import argparse
import sys
import os, copy
from utils import check_path
pjoin = os.path.join

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Farmland Extraction by Pytorch')
parser.add_argument('--root_path', metavar='DIR', help='path to dataset')
parser.add_argument('--dataset', help='dataset name')
parser.add_argument('-a', '--arch', metavar='ARCH', default='unet', help='model architecture: ' + ' | '.join(model_names) + ' (default: DSSA)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=64, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', '--batch_size', default=4, type=int, metavar='N', help='mini-batch size (default: 256), this is the total ' 'batch size of all GPUs on the current node when ' 'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--wd', '--weight_decay', default=1e-3, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('--milestones',  default=[35,45,55], help='scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [35,45,55], gamma=args.gamma)')
parser.add_argument('-p', '--print-freq', '--print_freq', default=20, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')

# routine params
parser.add_argument('--project_name', type=str, default="")
parser.add_argument('--debug', action="store_true")
parser.add_argument('--screen_print', action="store_true")
parser.add_argument('--note', type=str, default='', help='experiment note')
parser.add_argument('--print_interval', type=int, default=100)
parser.add_argument('--test_interval', type=int, default=2000)
parser.add_argument('--plot_interval', type=int, default=100000000)
parser.add_argument('--save_interval', type=int, default=2000, help="the interval to save model")
parser.add_argument('--ExpID', type=str, default='', help='Experiment id. In default it will be assigned automatically')
parser.add_argument('--save_init_model', action="store_true", help='save the model after initialization')


parser.add_argument('--gamma', '--gm', default=0.1, type=float, help='learning rate decay parameter: Gamma')
parser.add_argument('--itersize', default=1, help='iter size')
parser.add_argument("-plr", "--pretrainlr", type=float, default=0.1)
parser.add_argument('--loss_lmbda', default=1.1, type=float, help='hype-param of loss 1.1 for BSDS 1.3 for NYUD')
parser.add_argument("-num_classes", "--num_classes", type=int, default=1)
parser.add_argument('--LABELS',default='',help='LABELS without background')
parser.add_argument('--Colors',default='',help='RGB without background')
args = parser.parse_args()


args.dataset = 'Vaihingen_'
args.root_path = "/media/csu/ssd/xjwDLdatasets/Vaihingen_no_crop_v2"
# args.dataset =  'Potsdam_'
# args.root_path = '/media/csu/ssd/xjwDLdatasets/Potsdam_no_crop'

if args.dataset == 'Vaihingen_' or args.dataset == 'Potsdam_':

    args.LABELS = ["clutter","Impervious surfaces", "buildings", "low veg.", "trees", "cars"] # Label names
    args.Colors = {0:[255,0,0], 1: [255, 255, 255], 2: [0, 0, 255], 3: [0, 255, 255], 4: [0, 255, 0], 5: [255, 255, 0]}

args.num_classes = len(args.LABELS)

# args.arch = 'rsm_ss'
# args.arch = ''
args.arch = 'VPAMNet'
args.project_name = args.arch + '_' + args.dataset

args.epochs = 100
args.seed = 0
args.gpu = 0
# data
args.workers = 4
baseline_batch_size = 4
args.batch_size = 4
args.itersize = baseline_batch_size / args.batch_size
# opt
# args.LR = 0.001
args.LR = 6e-4
args.weight_decay = 0.01
args.momentum = 0.9
args.wd = 0.0005
args.milestones = [30,60]
args.gamma = 0.1
# args.solver = 'Adam'

args.print_freq = 20 * args.itersize
args.save_init_model = True
args.screen_print = True

# args.resume = r"D:\xjwdeeplearning\202407pesudo\exp220_seanet\checkpoint_best.pth"
# args.resume = r"D:\DLexp\202411\exp002_RSMamba_002_coslr_模板\Experiments\rsm_ss_gid15_SERVER-20241117-194119\weights\checkpoint_Last.pth"


args_tmp = {}
for k, v in args._get_kwargs():
    args_tmp[k] = v

# Above is the default setting. But if we explicitly assign new value for some arg in the shell script,
# the following will adjust the arg to the assigned value.
script = " ".join(sys.argv)
args.resume = check_path(args.resume)



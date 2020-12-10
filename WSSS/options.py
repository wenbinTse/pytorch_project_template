from argparse import ArgumentParser
import argparse
import os

def str2bool(opt: str):
    if opt.lower() in ['yes', '1', 'y', 'true']:
        return True
    elif opt.lower() in ['no', '0', 'n', 'false']:
        return False
    else:
        return argparse.ArgumentTypeError()

parser = ArgumentParser()

# distributed
parser.add_argument('--local_rank', type=int, default=-1)

# data
parser.add_argument('--dataset', type=str, default='voc', choices=['voc'])
parser.add_argument('--voc_root', type=str, default='/data/xiewenbin/VOC2012')
parser.add_argument('--save_dir', type=str, default='/data/xiewenbin/log/wsss/')

# model
parser.add_argument('--model', type=str, default='resnet50')

# training
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_scheduler', type=str, default='cos', choices=['cos', 'exp'])
parser.add_argument('--loss', type=str, default='1*ce', help='format: weight1*loss1+weight2*loss2')
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam'])
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--epochs', type=int, default=100)

parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--seed', type=int, default=666)
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)

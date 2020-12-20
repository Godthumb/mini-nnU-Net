import argparse
import torch.nn as nn
import torch
import logging
from file_and_folder_operations import join, exists, makedirs
import sys
from model import VNet
from dataset import COVID
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, default=r'D:/preprocessed_COVID19/resample_normalization',
                    help='resample data root dir')
parser.add_argument('--base_lr', type=float, default=0.01, help='initial lr')
parser.add_argument('--max_epoch', type=int, default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=2, help='batch_size per gpu')
parser.add_argument('--gpus', type=str, default='0', help='gpu to use')
parser.add_argument('--exp', type=str,  default='vnet_deep_supervised', help='model_name')
parser.add_argument('--patch_size', type=tuple, default=(128, 128, 128), help='patch_size')
parser.add_argument('--n_classes', type=int, default=2, help='seg classes')
parser.add_argument('--num_workers', type=int, default=4, help='how many thread used in Dataloader')
args = parser.parse_args()

snap_shot_path = './snap_shot/' + args.exp
batch_size = args.batch_size * len(args.gpus.split(','))
aug_dict = {'do_flip': True, 'do_swap': False}
def main():
    # make dir for snap_shot
    if not exists(snap_shot_path):
        makedirs(snap_shot_path)
    # logging
    logging.basicConfig(filename=join(snap_shot_path, 'log.txt'), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    # build model and init parameters with kaiming_he_normal_
    model = VNet(1, args.n_classes, 16, 'instancenorm', False)
    model.cuda()
    # make train / valid dataset and dataloader
    train_dataset = COVID(args.root_dir, args.patch_size, mode='train', **aug_dict)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=args.num_workers)

    val_dataset = COVID(args.root_dir, args.patch_size, mode='val', **aug_dict)

    val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                                  shuffle=False, num_workers=args.num_workers)


if __name__ == '__main__':
    main()
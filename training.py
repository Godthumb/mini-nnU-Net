import argparse
import torch.nn as nn
import torch
import logging
from file_and_folder_operations import join, exists, makedirs
import sys
from model import VNet
from dataset import COVID
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch.optim as optim
from tqdm import tqdm
import time
from losses import DC_and_CE_loss

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, default=r'D:/preprocessed_COVID19/resample_normalization',
                    help='resample data root dir')
parser.add_argument('--base_lr', type=float, default=0.01, help='initial lr')
parser.add_argument('--max_iterations', type=int,  default=50000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=2, help='batch_size per gpu')
parser.add_argument('--gpus', type=str, default='0', help='gpu to use')
parser.add_argument('--exp', type=str,  default='vnet_deep_supervised', help='model_name')
parser.add_argument('--patch_size', type=tuple, default=(64, 128, 128), help='patch_size')
parser.add_argument('--n_classes', type=int, default=2, help='seg classes')
parser.add_argument('--num_workers', type=int, default=4, help='how many thread used in Dataloader')
args = parser.parse_args()

snapshot_path = './snap_shot/' + args.exp
batch_size = args.batch_size * len(args.gpus.split(','))
aug_dict = {'do_flip': True, 'do_swap': False}

def main():
    # make dir for snap_shot
    if not exists(snapshot_path):
        makedirs(snapshot_path)
    # logging
    logging.basicConfig(filename=join(snapshot_path, 'log.txt'), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    # build model and init parameters with kaiming_he_normal_
    model = VNet(1, args.n_classes, 16, 'instancenorm', False)
    model = model.cuda()
    # make train / valid dataset and dataloader
    train_dataset = COVID(args.root_dir, args.patch_size, mode='train', aug_dict=aug_dict)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=args.num_workers)

    val_dataset = COVID(args.root_dir, args.patch_size, mode='val', aug_dict=aug_dict)

    val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                                  shuffle=False, num_workers=args.num_workers)


    # plot training produre
    writer = SummaryWriter(join(snapshot_path, 'log'), flush_secs=2)
    logging.info("{} itertations per epoch".format(len(train_dataloader)))

    # model in 'train' mode
    model.train()
    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    # weight
    weight_of_per_stage = [pow(1 / 2, i) for i in range(4)]  # [1, 1 /2, 1/4, 1/ 8]
    # dc_ce_loss
    dc_ce_loss = DC_and_CE_loss(smooth=1e-5, classes=args.n_classes, weight_ce=1, weight_dice=1)
    # print(weight_of_per_stage)
    iter_num = 0
    max_epoch = args.max_iterations // len(train_dataloader) + 1
    lr_ = args.base_lr
    # print(next(model.parameters()).is_cuda)
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for i, sample_data in enumerate(train_dataloader):
            time2 = time.time()
            # forward
            img, seg = sample_data
            img, seg = img.cuda(), seg.cuda()
            # print(torch.unique(seg))
            out_stage1, out_stage2, out_stage3, out_stage4 = model(img)
            loss = weight_of_per_stage[0] * dc_ce_loss(out_stage4, seg) +\
                            weight_of_per_stage[1] * dc_ce_loss(out_stage3, seg)+\
                            weight_of_per_stage[2] * dc_ce_loss(out_stage2, seg) +\
                            weight_of_per_stage[3] * dc_ce_loss(out_stage1, seg) # cuda tensor,value
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_num += 1
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))
            if iter_num % 2 == 0:
                pass

            if iter_num % 2500 == 0:
                lr_ = args.base_lr * 0.1 ** (iter_num // 2500) # (1e-3) ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            if iter_num % 5000 == 0:
                save_mode_path = join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num > args.max_iterations:
                break
            time1 = time.time()

        if iter_num > args.max_iterations:
            break

    save_mode_path = join(snapshot_path, 'iter_' + str(args.max_iterations + 1) + '.pth')
    torch.save(model.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    # close log write
    writer.close()




if __name__ == '__main__':
    main()
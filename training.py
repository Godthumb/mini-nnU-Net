import argparse
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
from losses import DC_and_CE_loss
import numpy as np
from losses import MultipleOutputLoss2
# from predict import calculate_metric_percase
import medpy.metric as metric

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, default=r'D:\COVID-19-20\preprocessed_COVID19',
                    help='resample data root dir')
parser.add_argument('--base_lr', type=float, default=0.01, help='initial lr')
parser.add_argument('--max_iterations', type=int,  default=50000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=2, help='batch_size per gpu')
parser.add_argument('--gpus', type=str, default='0', help='gpu to use')
parser.add_argument('--exp', type=str,  default='vnet_deep_supervised_COVID', help='model_name')
parser.add_argument('--patch_size', type=tuple, default=(64, 128, 128), help='patch_size')
parser.add_argument('--n_classes', type=int, default=2, help='seg classes')
parser.add_argument('--num_workers', type=int, default=4, help='how many thread used in Dataloader')
parser.add_argument('--has_deepsurpervised', type=bool, default=True, help='use deepsurpervised in highres layers')
args = parser.parse_args()

snapshot_path = './snap_shot/' + args.exp
batch_size = args.batch_size * len(args.gpus.split(','))
aug_dict = {'do_flip': True, 'do_swap': True}

def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)
    return dice, jc, hd, asd

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

    print(len(val_dataloader))

    # plot training produre
    writer = SummaryWriter(join(snapshot_path, 'log'), flush_secs=2)
    logging.info("{} itertations per epoch".format(len(train_dataloader)))

    # model in 'train' mode
    model.train()
    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    # weight
    num_pool = 4
    weights = np.array([pow(1 / 2, i) for i in range(num_pool)])  # [1, 1 /2, 1/4, 1/ 8]
    # use mask to control has_deepsurpervised
    if args.has_deepsurpervised:
        mask = np.array([True] + [True if i < num_pool - 1 else False for i in range(1, num_pool)])
    else:
        mask = np.array([True] + [False for i in range(1, num_pool)])
    weights[~mask] = 0
    print(weights)
    loss_weights = weights / weights.sum()
    print(loss_weights)
    # dc_ce_loss
    dc_ce_loss = DC_and_CE_loss(smooth=1e-5, classes=args.n_classes, weight_ce=1, weight_dice=1, ignore_classes_index=0)
    deep_supervised_loss = MultipleOutputLoss2(dc_ce_loss, loss_weights[::-1])
    iter_num = 0
    max_epoch = args.max_iterations // len(train_dataloader) + 1
    lr_ = args.base_lr
    # print(next(model.parameters()).is_cuda)
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        for i, sample_data in enumerate(train_dataloader):
            # forward
            img, seg = sample_data['image'], sample_data['label']
            img, seg = img.cuda(), seg.cuda()
            # print(torch.unique(seg))
            # out_stage1, out_stage2, out_stage3, out_stage4 = model(img)
            # loss = weight_of_per_stage[0] * dc_ce_loss(out_stage4, seg) +\
            #                 weight_of_per_stage[1] * dc_ce_loss(out_stage3, seg)+\
            #                 weight_of_per_stage[2] * dc_ce_loss(out_stage2, seg) +\
            #                 weight_of_per_stage[3] * dc_ce_loss(out_stage1, seg) # cuda tensor,value
            outputs = model(img)
            loss = deep_supervised_loss(outputs, seg)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_num += 1
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))
            # if iter_num % 2 == 0:
            #     model.eval()
            #     total_metric = 0.0
            #     with torch.no_grad():
            #         for img, seg in val_dataloader:
            #             img = img.cuda()
            #             seg = seg.numpy()
            #             output = model(img)[-1].cpu().numpy()
            #             output = np.argmax(output, axis=1)  # (b, 64, 128, 128)
            #             total_metric += np.asarray(calculate_metric_percase(output, seg[:, 0]))
            #     avg_metric = total_metric / (len(val_dataloader) * args.batch_size)
            #     print(avg_metric)
            #     model.train()

            if iter_num % 2500 == 0:
                lr_ = args.base_lr * 0.1 ** (iter_num // 2500) # (1e-3) ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            if iter_num % 250 == 0:
                save_mode_path = join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num > args.max_iterations:
                break

        # val every two epoch
        # convert to eval model
        if epoch_num % 2 == 0:
            model.eval()
            total_metric = 0.0
            with torch.no_grad():
                for sample_data in val_dataloader:
                    img = sample_data['image'].cuda()
                    seg = sample_data['label'].numpy()[:, 0]
                    output = model(img)[-1].cpu().numpy()
                    output = np.argmax(output, axis=1)  # (b, 64, 128, 128)
                    assert output.shape == seg.shape
                    total_metric += np.asarray(calculate_metric_percase(output, seg))
            avg_metric = total_metric / (len(val_dataloader) * args.batch_size)
            logging.info('dice: {}, jc: {}, hd95: {}, asd: {}'.format(avg_metric[0], avg_metric[1],
                                                                      avg_metric[2],
                                                                      avg_metric[3]))
            # change back to train model
            model.train()

        # save this epoch
        # save_mode_path = join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
        # torch.save(model.state_dict(), save_mode_path)
        # logging.info("save model to {}".format(save_mode_path))

        if iter_num > args.max_iterations:
            break

    save_mode_path = join(snapshot_path, 'iter_' + str(args.max_iterations + 1) + '.pth')
    torch.save(model.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    # close log write
    writer.close()




if __name__ == '__main__':
    main()

import argparse
from file_and_folder_operations import exists, makedirs, join, subfiles, load_pickle
from model import VNet
import torch
import numpy as np
import math
import torch.nn.functional as F
from medpy import metric
import SimpleITK as sitk
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./preprocessed/val', help='Name of experiment')
parser.add_argument('--model', type=str, default='vnet_deep_supervised', help='model_name')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--n_classes', type=int, default=2, help='seg classes')
args = parser.parse_args()

snapshot_path = './snap_shot/' + args.model
test_save_path = './predition/' + args.model + '_post'
if not exists(test_save_path):
    makedirs(test_save_path)

image_list = subfiles('./preprocessed/val', prefix=None, suffix='.npy', sort=False)

def get_caseidentifier(case):
    case_identifier = case.split('\\')[-1][:-4]
    return case_identifier


def test_calculate_metric(epoch_num, data_properties):
    model = VNet(n_channels=1, n_classes=args.n_classes, normalization='instancenorm', has_dropout=False).cuda()
    save_model_path = join(snapshot_path, 'iter_' + str(epoch_num) + '.pth')
    model.load_state_dict(torch.load(save_model_path))
    print("init weight from {}".format(save_model_path))
    target_spacing = load_pickle(data_properties)['target_spacing']
    model.eval()
    with torch.no_grad():
        avg_metric = test_all_case(model, image_list, 2, (64, 128, 128), 32, 64, save_result=True,
                               test_save_path=test_save_path, target_spacing=target_spacing)
    return avg_metric

def test_all_case(model, image_list, num_classes, patch_size=(64, 128, 128), stride_xy=32, stride_z=64,
                  save_result=True, test_save_path=None, target_spacing=None):

    total_metric = 0.0
    metric_dict = dict()
    metric_dict['name'] = list()
    metric_dict['dice'] = list()
    metric_dict['jaccard'] = list()
    metric_dict['asd'] = list()
    metric_dict['95hd'] = list()

    for image_path in image_list:
        case_name = get_caseidentifier(image_path)
        img, seg = np.load(image_path)  # (2, 234, 513, 513)
        prediction, score_map = test_single_case(model, img, 32, 64, (64, 128, 128), num_classes)
        if np.sum(prediction) == 0:
            single_metric = (0, 0, 0, 0)
        else:
            single_metric = calculate_metric_percase(prediction, seg)
            metric_dict['name'].append(case_name)
            metric_dict['dice'].append(single_metric[0])
            metric_dict['jaccard'].append(single_metric[1])
            metric_dict['asd'].append(single_metric[2])
            metric_dict['95hd'].append(single_metric[3])

        total_metric += np.asarray(single_metric)
        if save_result:
            test_save_path_temp = join(test_save_path, case_name)
            if not exists(test_save_path_temp):
                makedirs(test_save_path_temp)
                pred_sitk = sitk.GetImageFromArray(prediction.astype(np.float32))
                pred_sitk.SetSpacing(target_spacing[::-1])
                sitk.WriteImage(pred_sitk, test_save_path_temp + '/' + case_name + "_pred.nii.gz")

                img_sitk = sitk.GetImageFromArray(img[:].astype(np.float32))
                img_sitk.SetSpacing(target_spacing[::-1])
                sitk.WriteImage(img_sitk, test_save_path_temp + '/' + case_name + "_img.nii.gz")

                gt_sitk = sitk.GetImageFromArray(seg[:].astype(np.float32))
                gt_sitk.SetSpacing(target_spacing[::-1])
                sitk.WriteImage(gt_sitk, test_save_path_temp + '/' + case_name + "_gt.nii.gz")

    avg_metric = total_metric / len(image_list)
    metric_csv = pd.DataFrame(metric_dict)
    metric_csv.to_csv(test_save_path + '/metric.csv', index=False)
    print('average metric is {}'.format(avg_metric))

# stride_xy = 1 / 2 * patch_size_xy
# stride_z = 1 /2 * patch_size_z
def test_single_case(model, image, stride_xy=64, stride_z=32, patch_size=(64, 128, 128), num_classes=2):
    d, h, w = image.shape
    # if size of image is less than patch_size, then padding it
    add_pad = False
    if d < patch_size[0]:
        d_pad = patch_size[0] - d
        add_pad = True
    else:
        d_pad = 0

    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0

    if w < patch_size[2]:
        w_pad = patch_size[2] - w
        add_pad = True
    else:
        w_pad = 0

    # pad on both side
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    if add_pad:
        image = np.pad(image, ((dl_pad, dr_pad), (hl_pad, hr_pad), (wl_pad, wr_pad)), mode='constant',
                       constant_values=0)

    dd, hh, ww = image.shape
    sz = math.ceil((dd - patch_size[0]) / stride_z) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sx = math.ceil((ww - patch_size[2]) / stride_xy) + 1
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32) # (2, d, h, w)
    cnt = np.zeros(image.shape).astype(np.float32)

    for z in range(0, sz):
        zs = min(stride_z * z, dd - patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for x in range(0, sx):
                xs = min(stride_xy * x, ww - patch_size[2])
                test_patch = image[zs: zs + patch_size[0], ys: ys + patch_size[1], xs: xs + patch_size[2]]
                # (64, 128, 128) -> (1, 1, 64, 128, 128)
                test_patch = np.expand_dims(np.expand_dims(test_patch, axis=0), axis=0)
                test_patch = torch.from_numpy(test_patch).cuda()
                y1 = model(test_patch)[-1]  # select last out
                y = F.softmax(y1, dim=1) # softmax by c
                y = y.cpu().detach().numpy()
                y = y[0, ...] # (1, 2, 64, 128, 128) -> (2, 64, 128, 128)
                score_map[:, zs: zs + patch_size[0], ys: ys + patch_size[1], xs: xs + patch_size[2]] += y
                cnt[zs: zs + patch_size[1], ys: ys + patch_size[2], xs: xs + patch_size[2]] += 1

    score_map = score_map / np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)  # (64, 128, 128)

    if add_pad:
        label_map = label_map[dl_pad:dl_pad + d, hl_pad:hl_pad + h, wl_pad:wl_pad + w]
        score_map = score_map[dl_pad:dl_pad + d, hl_pad:hl_pad + h, wl_pad:wl_pad + w]
    return label_map, score_map

def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt) # 95th percentile of the Hausdorff Distance.
    asd = metric.binary.asd(pred, gt) # Average surface distance metric.
    return dice, jc, hd, asd


if __name__ == '__main__':
    metric = test_calculate_metric(50001, './preprocessed/foreground/dataset_properties.pkl')

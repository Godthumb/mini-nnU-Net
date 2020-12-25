from torch.utils.data import Dataset
import numpy as np
from file_and_folder_operations import subfiles, join
import pickle
import random

class COVID(Dataset):
    def __init__(self, base_dir, patch_size, batch_size=2, mode='train', oversample_foreground_percent=0.33, aug_dict=None):
        super(COVID, self).__init__()
        assert mode == 'train' or mode == 'val'
        self.base_dir = join(base_dir, mode)
        self.resample_data_file_list = subfiles(self.base_dir, None, '.npy', False) # npy files
        # print(self.resample_data_file_list)
        # shuffle here
        random.shuffle(self.resample_data_file_list)
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.oversample_foreground_percent = oversample_foreground_percent
        self.mode = mode
        self.aug_dict = aug_dict

    def __getitem__(self, idx):
        this_case = self.resample_data_file_list[idx]
        # print(idx% self.batch_size)
        if self.get_do_oversample(idx% self.batch_size):
            force_fg = True
        else:
            force_fg = False
        # print(force_fg)
        case_identifier = self.get_case_identifier(this_case)
        data, seg, properties = self.load_all_data(case_identifier) # # (1, 234, 512, 512)
        case_all_data = np.concatenate([data, seg], axis=0)
        need_to_pad = [0, 0, 0]
        # compute lb & ub
        shape = case_all_data.shape[1:]  # z, y, x
        for d in range(3):
            if shape[d] < self.patch_size[d]:  # for z axis here
                need_to_pad[d] = self.patch_size[d] - shape[d]

        lb_x = -need_to_pad[2] // 2
        ub_x = shape[2] + need_to_pad[2] // 2 + need_to_pad[2] % 2 - self.patch_size[2]
        lb_y = -need_to_pad[1] // 2
        ub_y = shape[1] + need_to_pad[1] // 2 + need_to_pad[1] % 2 - self.patch_size[1]
        lb_z = -need_to_pad[0] // 2
        ub_z = shape[0] + need_to_pad[0] // 2 + need_to_pad[0] % 2 - self.patch_size[0]
        if not force_fg:
            bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
            bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
            bbox_z_lb = np.random.randint(lb_z, ub_z + 1)

        else:
            # if foreground pixel exist, use choosed foreground pixel random sample slice
            foreground_classes = np.array(
                [i for i in properties['class_locations'].keys() if len(properties['class_locations'][i]) != 0])
            foreground_classes = foreground_classes[foreground_classes > 0]
            if len(foreground_classes) == 0:
                selected_class = None
                voxels_of_that_class = None
                print('case does not contain any foreground classes', idx)
            else:
                selected_class = np.random.choice(foreground_classes)
                voxels_of_that_class = properties['class_locations'][selected_class]

            if voxels_of_that_class is None:
                bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
                bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
                bbox_z_lb = np.random.randint(lb_z, ub_z + 1)
            else:
                selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
                bbox_x_lb = max(lb_x, selected_voxel[2] - self.patch_size[2] // 2)
                bbox_y_lb = max(lb_y, selected_voxel[1] - self.patch_size[1] // 2)
                bbox_z_lb = max(lb_z, selected_voxel[0] - self.patch_size[0] // 2)

        bbox_x_ub = bbox_x_lb + self.patch_size[2]
        bbox_y_ub = bbox_y_lb + self.patch_size[1]
        bbox_z_ub = bbox_z_lb + self.patch_size[0]

        # get valid box within img
        valid_bbox_x_lb = max(0, bbox_x_lb)
        valid_bbox_x_ub = min(shape[2], bbox_x_ub)
        valid_bbox_y_lb = max(0, bbox_y_lb)
        valid_bbox_y_ub = min(shape[1], bbox_y_ub)
        valid_bbox_z_lb = max(0, bbox_z_lb)
        valid_bbox_z_ub = min(shape[0], bbox_z_ub)
        # select voxel in valid box
        case_all_data = np.copy(case_all_data[:, valid_bbox_z_lb:valid_bbox_z_ub,
                                valid_bbox_y_lb:valid_bbox_y_ub,
                                valid_bbox_x_lb:valid_bbox_x_ub])

        # pad valid voxel to patch size
        patch_data = np.pad(case_all_data[:-1], ((0, 0),
                                    (-min(0, bbox_z_lb), max(bbox_z_ub - shape[0], 0)),
                                    (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                                    (-min(0, bbox_x_lb), max(bbox_x_ub - shape[2], 0))), mode='constant', constant_values=0)

        patch_seg = np.pad(case_all_data[-1:], ((0, 0),
                                                 (-min(0, bbox_z_lb), max(bbox_z_ub - shape[0], 0)),
                                                 (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                                                 (-min(0, bbox_x_lb), max(bbox_x_ub - shape[2], 0))),
                           mode='constant', constant_values=0)
        
        # do_augmentation while training phase
        # if self.mode == 'train':
        #     patch_data, patch_seg = self.do_augment(patch_data, patch_seg, **self.aug_dict)
        return {'image': patch_data, 'label': patch_seg}

    def do_augment(self, patch_data, patch_seg, do_flip=True, do_swap=True, do_rot=True):
        if do_flip:
            flip_id = np.array([1, np.random.randint(2), np.random.randint(2)]) * 2 - 1 # [1, -1, -1]
            patch_data = np.ascontiguousarray(patch_data[:, ::flip_id[0], ::flip_id[1], ::flip_id[2]])
            patch_seg = np.ascontiguousarray(patch_seg[:, ::flip_id[0], ::flip_id[1], ::flip_id[2]])

        if do_swap:
            if patch_data.shape[1] == patch_data.shape[3] and patch_data.shape[1] == patch_data.shape[2]:
                axisorder = np.random.permutation(3) # [1, 0, 2]
                patch_data = np.transpose(patch_data, np.concatenate([[0], axisorder + 1]))
                patch_seg = np.transpose(patch_seg, np.concatenate([[0], axisorder + 1]))
        if do_rot:
            k = np.random.randint(0, 4)
            patch_data = np.rot90(patch_data, k, axes=(1, 2))
            patch_seg = np.rot90(patch_seg, k, axes=(1, 2))

        return patch_data, patch_seg

    def get_do_oversample(self, idx):  # 0.67
        # 2 * (1- 0.33) = round(2 * 0.67) = 1
        return idx < round(self.batch_size * (1 - self.oversample_foreground_percent))

    def get_case_identifier(self, case):
        case_identifier = case.split('\\')[-1][:-4]
        return case_identifier

    # get data & seg & properties
    def load_all_data(self, case_identifier):
        all_data = np.load(join(self.base_dir, "%s.npy" % case_identifier))
        data = all_data[:-1].astype(np.float32) # (1, 234, 512, 512)
        seg = all_data[-1:]  # (1, 234, 512, 512)
        with open(join(self.base_dir, "%s.pkl" % case_identifier), 'rb') as f:
            properties = pickle.load(f)
        return data, seg, properties


    def __len__(self):
        return len(self.resample_data_file_list)




if __name__ == '__main__':
    import SimpleITK as sitk
    from file_and_folder_operations import *
    data = load_pickle(r'./preprocessed/foreground/dataset_properties.pkl')['target_spacing']
    print(data)
    dataset = COVID(r'./preprocessed', (64, 128, 128), 'train', {'do_flip': True, 'do_swap': False})
    preprocess_dict = dataset.__getitem__(1)
    img, seg = preprocess_dict['image'], preprocess_dict['label']
    print(img.shape)
    print(seg.shape)
    print(np.unique(img))
    print(np.unique(seg))
    # img, seg = dataset.do_augment(img, seg, {'do_flip': True, 'do_swap': False})
    img = sitk.GetImageFromArray(img[0])
    img.SetSpacing(np.array(data)[::-1])
    seg = sitk.GetImageFromArray(seg[0])
    seg.SetSpacing(np.array(data)[::-1])
    sitk.WriteImage(img, './aug_img.nii.gz')
    sitk.WriteImage(seg, './aug_seg.nii.gz')


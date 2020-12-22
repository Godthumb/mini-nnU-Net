from torch.utils.data import Dataset
import numpy as np
from file_and_folder_operations import subfiles, join
import pickle

class COVID(Dataset):
    def __init__(self, base_dir, patch_size, mode='train', aug_dict=None):
        super(COVID, self).__init__()
        assert mode == 'train' or mode == 'val'
        self.base_dir = join(base_dir, mode)
        self.resample_data_file_list = subfiles(self.base_dir, None, '.npy', False) # npy files
        self.patch_size = patch_size
        self.mode = mode
        self.aug_dict = aug_dict

    def __getitem__(self, idx):
        this_case = self.resample_data_file_list[idx]
        case_identifier = self.get_case_identifier(this_case)
        data, seg, properties = self.load_all_data(case_identifier) # # (1, 234, 512, 512)
        case_all_data = np.concatenate([data, seg], axis=0)
        # if foreground pixel exist, use choosed foreground pixel random sample slice
        foreground_classes = np.array(
            [i for i in properties['class_locations'].keys() if len(properties['class_locations'][i]) != 0])
        foreground_classes = foreground_classes[foreground_classes > 0]
        if len(foreground_classes) == 0:
            selected_class = None
            print('case does not contain any foreground classes', idx)
        else:
            selected_class = np.random.choice(foreground_classes)
            voxels_of_that_class = properties['class_locations'][selected_class]

        # compute lb & ub
        shape = case_all_data.shape[1:] # z, y, x
        lb_x = 0
        ub_x = shape[2] - self.patch_size[2]
        lb_y = 0
        ub_y = shape[1] - self.patch_size[1]
        lb_z = 0
        ub_z = shape[0] - self.patch_size[0]

        if selected_class is None:
            bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
            bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
            bbox_z_lb = np.random.randint(lb_z, ub_z + 1)
        else:
            selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
            # print(selected_voxel)
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
        if self.mode == 'train':
            patch_data, patch_seg = self.do_augment(patch_data, patch_seg, **self.aug_dict)
        return {'image': patch_data, 'label': patch_seg}

    def do_augment(self, patch_data, patch_seg, do_flip=True, do_swap=True):
        if do_flip:
            flip_id = np.array([1, np.random.randint(2), np.random.randint(2)]) * 2 - 1 # [1, -1, -1]
            patch_data = np.ascontiguousarray(patch_data[:, ::flip_id[0], ::flip_id[1], ::flip_id[2]])
            patch_seg = np.ascontiguousarray(patch_seg[:, ::flip_id[0], ::flip_id[1], ::flip_id[2]])

        if do_swap:
            if patch_data.shape[1] == patch_data.shape[3] and patch_data.shape[1] == patch_data.shape[2]:
                axisorder = np.random.permutation(3) # [1, 0, 2]
                patch_data = np.transpose(patch_data, np.concatenate([[0], axisorder + 1]))
                patch_seg = np.transpose(patch_seg, np.concatenate([[0], axisorder + 1]))


        return patch_data, patch_seg

    def get_do_oversample(self, p):  # 0.67
        # 2 * (1- 0.33) = round(2 * 0.67) = 1
        if np.random.uniform(0, 1) < p:
            return True
        else:
            return False

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
    from torch.utils.data import DataLoader
    dataset = COVID(r'D:\COVID-19-20\preprocessed_COVID19', (64, 128, 128), 'train', {'do_flip': True, 'do_swap': False})
    # img, seg = dataset.__getitem__(0)
    # print(img.shape)
    # print(seg.shape)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    from file_and_folder_operations import *
    # img, seg = dataset.do_augment(img, seg, )
    target_spacing = load_pickle(r'D:\COVID-19-20\preprocessed_COVID19\crop_foreground\dataset_properties.pkl')['target_spacing']
    for i, sample_data in enumerate(dataloader):
        img, seg = sample_data['image'], sample_data['label']
        print(img.shape)
        img = sitk.GetImageFromArray(img[0, 0, ...])
        img.SetSpacing(np.array(target_spacing)[::-1])
        seg = sitk.GetImageFromArray(seg[0, 0, ...])
        seg.SetSpacing(np.array(target_spacing)[::-1])
        sitk.WriteImage(img, './aug_img_%s.nii.gz' % str(i))
        sitk.WriteImage(seg, './aug_seg_%s.nii.gz' % str(i))

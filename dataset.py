from torch.utils.data import Dataset
import numpy as np
from file_and_folder_operations import subfiles, join
import pickle
import random
from scipy import ndimage

class COVID(Dataset):
    def __init__(self, base_dir, patch_size, batch_size=2, mode='train', oversample_foreground_percent=0.33, transform=None):
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
        self.transform = transform

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
        patch_data_before = np.pad(case_all_data[:-1], ((0, 0),
                                    (-min(0, bbox_z_lb), max(bbox_z_ub - shape[0], 0)),
                                    (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                                    (-min(0, bbox_x_lb), max(bbox_x_ub - shape[2], 0))), mode='constant', constant_values=0)

        patch_seg = np.pad(case_all_data[-1:], ((0, 0),
                                                 (-min(0, bbox_z_lb), max(bbox_z_ub - shape[0], 0)),
                                                 (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                                                 (-min(0, bbox_x_lb), max(bbox_x_ub - shape[2], 0))),
                           mode='constant', constant_values=0)


        sample = {'image': patch_data_before, 'label': patch_seg}
        # do_augmentation while training phase
        if self.mode == 'train' and self.transform is not None:
            sample = self.transform(sample)
        return sample

    def get_do_oversample(self, idx):  # 0.67
        # 4 * (1- 0.33) = round(4 * 0.67) = 3
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

class GaussianNoiseTransform(object):
    def __init__(self, p=0.1, noise_variance=(0, 0.1)):
        self.p = p
        self.noise_variance = noise_variance

    def __call__(self, sample):
        image, seg = sample['image'], sample['label']
        if np.random.uniform() < self.p:
            if self.noise_variance[0] == self.noise_variance[1]:
                variance = self.noise_variance[0]
            else:
                variance = random.uniform(self.noise_variance[0], self.noise_variance[1]) # (0, 0.1)
            image = image + np.random.normal(0.0, variance, size=image.shape)
            image = image.astype(np.float32)
        return {'image': image, 'label': seg}


class GaussianBlurTransform(object):
    def __init__(self, p=0.2, sigma_range=(0.5, 1)):
        self.p = p
        self.sigma = np.random.uniform(sigma_range[0], sigma_range[1])

    def __call__(self, sample):
        image, seg = sample['image'], sample['label']
        if np.random.uniform() < self.p:
            image = ndimage.gaussian_filter(image, self.sigma, order=0)

        return {'image': image, 'label': seg}


class GammaTransform(object):
    def __init__(self, p=0.3, gamma_range=(0.7, 1.5)):
        self.p = p
        self.gamma_range = gamma_range

    def __call__(self, sample):
        image, seg = sample['image'], sample['label']
        if np.random.uniform() < self.p:
            mn = image.mean()
            sd = image.std()
            if np.random.random() < 0.5 and self.gamma_range[0] < 1:
                gamma = np.random.uniform(self.gamma_range[0], 1)  # [0.7 ,1]
            else:
                gamma = np.random.uniform(max(self.gamma_range[0], 1), self.gamma_range[1])  # [1, 1.5]

            # (x - x.min()) / (x.max() - x.min()) => (0, 1)
            minm = image.min()
            rnge = image.max() - minm
            # gamma transform & transform back use minm, rnge
            image = np.power(((image - minm) / float(rnge + 1e-7)), gamma) * rnge + minm
            image = image - image.mean() + mn
            image = image / (image.std() + 1e-8) * sd

        return {'image': image, 'label': seg}


class RotationTransform(object):
    def __init__(self, p=0.2, rotate_range=(-30, 30)):
        self.p = p
        self.angle = random.randint(rotate_range[0], rotate_range[1])

    def __call__(self, sample):
        image, seg = sample['image'], sample['label']
        if np.random.uniform() < self.p:
            image = ndimage.rotate(image, self.angle, (2, 3), reshape=False, cval=0)
            seg = np.round(ndimage.rotate(seg, self.angle, (2, 3), reshape=False, cval=0))
        return {'image': image, 'label': seg}


class FlipTransform(object):
    def __call__(self, sample):
        image, seg = sample['image'], sample['label']
        flip_id = np.array([1, np.random.randint(2), np.random.randint(2)]) * 2 - 1  # [1, -1, -1]
        image = np.ascontiguousarray(image[:, ::flip_id[0], ::flip_id[1], ::flip_id[2]])
        seg = np.ascontiguousarray(seg[:, ::flip_id[0], ::flip_id[1], ::flip_id[2]])
        return {'image': image, 'label': seg}




if __name__ == '__main__':
    import SimpleITK as sitk
    from file_and_folder_operations import *
    import torchvision.transforms as transforms
    data = load_pickle(r'./preprocessed/foreground/dataset_properties.pkl')['target_spacing']
    print(data)
    transforms_ = transforms.Compose([FlipTransform(), RotationTransform(1),
                                      GaussianNoiseTransform(1), GaussianBlurTransform(1), GammaTransform(1)])
    dataset = COVID(r'./preprocessed', (64, 128, 128), 2, 'train', 0.33, transform=transforms_)
    preprocess_dict = dataset.__getitem__(0)
    aug_img, seg = preprocess_dict['image'], preprocess_dict['label']
    print(aug_img.dtype)
    print(seg.dtype)
    aug_img = sitk.GetImageFromArray(aug_img[0])
    aug_img.SetSpacing(np.array(data)[::-1])
    seg = sitk.GetImageFromArray(seg[0])
    seg.SetSpacing(np.array(data)[::-1])
    sitk.WriteImage(aug_img, './aug_img.nii.gz')
    sitk.WriteImage(seg, './aug_seg.nii.gz')



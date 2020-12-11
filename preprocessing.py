# class for resampling
# get target_spacing & resampling
import numpy as np
from batchgenerators.augmentations.utils import resize_segmentation
from skimage.transform import resize
# import pickle
# import os
from scipy.ndimage.interpolation import map_coordinates
from file_and_folder_operations import *
from multiprocessing import Pool
# def load_pickle(file, mode='rb'):
#     with open(file, mode) as f:
#         a = pickle.load(f)
#     return a
#
# join = os.path.join

def subfiles(folder, prefix=None, suffix=None, sort=True):
    l = os.path.join
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isfile(os.path.join(folder, i))
           and (prefix is None or i.startswith(prefix))
           and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res

def get_case_identifier(case):
    case_identifier = case.split('\\')[-1][:-4]
    return case_identifier

def get_do_separate_z(spacing, anisotropy_threshold=3):
    do_separate_z = (np.max(spacing) / np.min(spacing)) > anisotropy_threshold
    return do_separate_z

# return [0]
def get_lowres_axis(new_spacing):
    axis = np.where(max(new_spacing) / np.array(new_spacing) == 1)[0]  # find which axis is anisotropic
    return axis

def resample_data_or_seg(data, new_shape, is_seg, axis=None, order=3, do_separate_z=False, cval=0, order_z=0):
    """
     separate_z=True will resample with order 0 along z
     :param data:
     :param new_shape:
     :param is_seg:
     :param axis:
     :param order:
     :param do_separate_z:
     :param cval:
     :param order_z: only applies if do_separate_z is True
     :return:
     """
    assert len(data.shape) == 4, "data must be (c, x, y, z)"
    if is_seg:
        resize_fn = resize_segmentation
        kwargs = dict()
    else:
        resize_fn = resize
        kwargs = {'mode': 'edge', 'anti_aliasing': False}

    dtype_data = data.dtype
    data = data.astype(float)
    shape = np.array(data[0].shape)  # (234, 512, 512)
    new_shape = np.array(new_shape)  #
    if np.any(shape != new_shape):
        if do_separate_z:
            print("separate z, order in z is", order_z, "order inplane is", order)
            assert len(axis) == 1, "only one anisotropic axis supported"
            axis = axis[0]  # should be 0
            new_shape_2d = new_shape[1:] # (512, 512)
            # reshaped_final_data = []
            reshaped_data = []
            # travel over each slice, and do x, y resample
            for slice_id in range(shape[0]):
                reshaped_data.append(resize_fn(data[0, slice_id, :, :], new_shape_2d, order, cval=cval, **kwargs))
            reshaped_data = np.stack(reshaped_data, axis=axis)
            # do z resample
            if shape[axis] != new_shape[axis]:
                # The following few lines are blatantly copied and modified from sklearn's resize()
                dim, cols, rows = new_shape[0], new_shape[1], new_shape[2]  # z, y, x    (360, 520, 520)
                orig_dim, orig_cols, orig_rows = reshaped_data.shape  # (234, 520, 520)

                dim_scale = float(orig_dim) / dim
                col_scale = float(orig_cols) / cols
                row_scale = float(orig_rows) / rows

                map_dims, map_cols, map_rows = np.mgrid[:dim, :cols, :rows]

                map_dims = dim_scale * (map_dims + 0.5) - 0.5
                map_cols = col_scale * (map_cols + 0.5) - 0.5
                map_rows = row_scale * (map_rows + 0.5) - 0.5

                coord_map = np.array([map_dims, map_cols, map_rows])
                # resample img
                if not is_seg or order_z == 0:
                    reshaped_final_data = map_coordinates(reshaped_data, coord_map, order=order_z, cval=cval,
                                                               mode='nearest')[None]  # (1, 360, 520, 520)
                # resample seg
                else:
                    unique_labels = np.unique(reshaped_data)
                    reshaped = np.zeros(new_shape, dtype=dtype_data)

                    for i, cl in enumerate(unique_labels):
                        reshaped_multihot = np.round(
                            map_coordinates((reshaped_data == cl).astype(float), coord_map, order=order_z,
                                            cval=cval, mode='nearest'))
                        reshaped[reshaped_multihot > 0.5] = cl
                    reshaped_final_data = reshaped[None]
            else:
                reshaped_final_data = reshaped_data[None]

        else:
            print("no separate z, order", order)
            # reshaped = []
            # for c in range(data.shape[0]):
            reshaped_final_data = resize_fn(data[0], new_shape, order, cval=cval, **kwargs)[None]
            # reshaped_final_data = np.vstack(reshaped)
        return reshaped_final_data.astype(dtype_data)

    else:
        print("no resampling necessary")
        return data

def resample_patient(data, seg, original_spacing, target_spacing, order_data=3, order_seg=0,
                     cval_data=0, cval_seg=-1, order_z_data=0, order_z_seg=0,
                     separate_z_anisotropy_threshold=3):
    """
    :param cval_seg:
    :param cval_data:
    :param data:
    :param seg:
    :param original_spacing:
    :param target_spacing:
    :param order_data:
    :param order_seg:
    :param force_separate_z: if None then we dynamically decide how to resample along z, if True/False then always
    /never resample along z separately
    :param order_z_seg: only applies if do_separate_z is True
    :param order_z_data: only applies if do_separate_z is True
    :param separate_z_anisotropy_threshold: if max_spacing > separate_z_anisotropy_threshold * min_spacing (per axis)
    then resample along lowres axis with order_z_data/order_z_seg instead of order_data/order_seg

    :return:
    """
    assert not ((data is None) and (seg is None))

    if data is not None:
        assert len(data.shape) == 4, "data must be c z y x"
    if seg is not None:
        assert len(seg.shape) == 4, "seg must be c z y x"

    if data is not None:
        shape = np.array(data[0].shape)
    else:
        shape = np.array(seg[0].shape)
    new_shape = np.round(((np.array(original_spacing) / np.array(target_spacing)).astype(float) * shape)).astype(int)

    # determine is to do_separate_z or not
    # step 1: original_spacing or target_spacing is anisotropy or not
    # step 2: not anisotropy
    if get_do_separate_z(original_spacing, separate_z_anisotropy_threshold):
        do_separate_z = True
        axis = get_lowres_axis(original_spacing) # [0]
    elif get_do_separate_z(target_spacing, separate_z_anisotropy_threshold):
        do_separate_z = True
        axis = get_lowres_axis(target_spacing)
    else:
        do_separate_z = False
        axis = None

    # resample img
    data_reshaped = resample_data_or_seg(data, new_shape, False, axis, order_data, do_separate_z, cval=cval_data,
                                         order_z=order_z_data)
    # resample seg
    seg_reshaped = resample_data_or_seg(seg, new_shape, True, axis, order_seg, do_separate_z, cval=cval_seg,
                                        order_z=order_z_seg)
    return data_reshaped, seg_reshaped

class GenericPreprocessor(object):
    def __init__(self, folder_with_cropped_data, out_dir, num_thread=4):
        # self.transpose_forward = transpose_forward

        self.resample_separate_z_anisotropy_threshold = 3
        self.out_dir = out_dir
        # self.use_nonzero_mask = False # False in CT data
        # get data pickle
        self.folder_with_cropped_data = folder_with_cropped_data
        self.intensityproperties = self.load_dataset_properties(folder_with_cropped_data)
        self.dataset_properties = load_pickle(join(self.folder_with_cropped_data, "dataset_properties.pkl"))
        self.target_spacing_percentile = 50
        self.num_thread = num_thread

    @staticmethod
    def load_cropped(cropped_output_dir, case_identifier):
        all_data = np.load(join(cropped_output_dir, "%s.npy" % case_identifier))
        data = all_data[:-1].astype(np.float32) # (1, 234, 512, 512)
        seg = all_data[-1:]  # (1, 234, 512, 512)
        with open(join(cropped_output_dir, "%s.pkl" % case_identifier), 'rb') as f:
            properties = pickle.load(f)
        return data, seg, properties

    @staticmethod
    def load_dataset_properties(cropped_output_dir):
        with open(join(cropped_output_dir, 'dataset_properties.pkl'), 'rb') as f:
            dataset_properties = pickle.load(f)

        return dataset_properties['intensityproperties']

    def get_spacings(self):
        spacings = self.dataset_properties['all_spacings']
        sizes = self.dataset_properties['all_sizes']

        target = np.percentile(np.vstack(spacings), self.target_spacing_percentile, 0)

        # This should be used to determine the new median shape. The old implementation is not 100% correct.
        # Fixed in 2.4
        # sizes = [np.array(i) / target * np.array(j) for i, j in zip(spacings, sizes)]

        target_size = np.percentile(np.vstack(sizes), self.target_spacing_percentile, 0)
        # we need to identify datasets for which a different target spacing could be beneficial. These datasets have
        # the following properties:
        # - one axis which much lower resolution than the others
        # - the lowres axis has much less voxels than the others
        # - (the size in mm of the lowres axis is also reduced)
        worst_spacing_axis = np.argmax(target)
        other_axes = [i for i in range(len(target)) if i != worst_spacing_axis]
        other_spacings = [target[i] for i in other_axes]
        other_sizes = [target_size[i] for i in other_axes]

        has_aniso_spacing = target[worst_spacing_axis] > (self.resample_separate_z_anisotropy_threshold * max(other_spacings))
        has_aniso_voxels = target_size[worst_spacing_axis] * self.resample_separate_z_anisotropy_threshold < min(other_sizes)

        if has_aniso_spacing and has_aniso_voxels:
            spacings_of_that_axis = np.vstack(spacings)[:, worst_spacing_axis]
            target_spacing_of_that_axis = np.percentile(spacings_of_that_axis, 10)
            # don't let the spacing of that axis get higher than the other axes
            if target_spacing_of_that_axis < max(other_spacings):
                target_spacing_of_that_axis = max(max(other_spacings), target_spacing_of_that_axis) + 1e-5
            target[worst_spacing_axis] = target_spacing_of_that_axis
        return target

    def resample_and_normalize(self, data, target_spacing, properties, case_identifier, seg=None):
        """
        data and seg must already have been transposed by transpose_forward. properties are the un-transposed values
        (spacing etc)
        :param data:
        :param target_spacing:
        :param properties:
        :param seg:
        :param force_separate_z:
        :return:
        """
        before = {
            'spacing': properties["original_spacing"],
            'data.shape (data is transposed)': data.shape
        }

        # remove nans
        data[np.isnan(data)] = 0
        # skimage.transform.resize()
        # The order of interpolation. The order has to be in the range 0-5:
        # 0: Nearest-neighbor
        # 1: Bi-linear (default)
        # 2: Bi-quadratic
        ####### 3: Bi-cubic
        # 4: Bi-quartic
        # 5: Bi-quintic
        ######################################################
        data, seg = resample_patient(data, seg, np.array(properties['original_spacing']), target_spacing, 3, 1
                                     , order_z_data=0, order_z_seg=0,
                                     separate_z_anisotropy_threshold=self.resample_separate_z_anisotropy_threshold)
        after = {
            'spacing': target_spacing,
            'data.shape (data is resampled)': data.shape
        }
        # print out spacing change and shape change on screen
        print("before:", before, "\nafter: ", after, "\n")

        if seg is not None:  # hippocampus 243 has one voxel with -2 as label. wtf?
            seg[seg < -1] = 0

        properties["size_after_resampling"] = data[0].shape
        properties["spacing_after_resampling"] = target_spacing
        # use_nonzero_mask = self.use_nonzero_mask
        #
        # CT normalization
        mean_intensity = self.intensityproperties['mean']
        std_intensity = self.intensityproperties['sd']
        lower_bound = self.intensityproperties['percentile_00_5']
        upper_bound = self.intensityproperties['percentile_99_5']
        # print(mean_intensity, std_intensity, lower_bound, upper_bound)
        # truncation by percentile_00_5 and percentile_99_5
        data[0] = np.clip(data[0], lower_bound, upper_bound)
        # z-score (img - mean) / std
        data[0] = (data[0] - mean_intensity) / std_intensity
        # if use_nonzero_mask[0]:
        #      data[0][seg[-1] < 0] = 0
        all_data = np.vstack((data, seg)).astype(np.float32)
        # return data, seg, properties
        np.save(join(self.out_dir, "%s" % case_identifier) + '.npy', all_data)
        with open(join(self.out_dir, "%s.pkl" % case_identifier), 'wb') as f:
            pickle.dump(properties, f)

    def run(self):
        target_spacings = self.get_spacings()
        # get all npy files is a list
        list_of_cropped_files = subfiles(self.folder_with_cropped_data, None, '.npy', True)
        all_args = []
        for j, case in enumerate(list_of_cropped_files):
            case_identifier = get_case_identifier(case)
            data, seg, property = self.load_cropped(self.folder_with_cropped_data, case_identifier)
            args = data, target_spacings, property, case_identifier, seg
            all_args.append(args)

        p = Pool(self.num_thread)
        p.starmap(self.resample_and_normalize, all_args)
        p.close()
        p.join()
        # for arg in all_args:
        #     self.resample_and_normalize(*arg)

if __name__ == '__main__':
    processer = GenericPreprocessor('D:/preprocessed_COVID19/crop_foreground',
                                    'D:/preprocessed_COVID19/resample_normalization',
                                    )
    processer.run()


from multiprocessing.pool import Pool
from utils import *
import numpy as np
import pickle
import SimpleITK as sitk

def get_case_identifier(case):
    case_identifier = case[0].split('\\')[-1].split('.nii.gz')[0][:-3]
    return case_identifier

def load_case_from_list_of_files(data_files, seg_file=None):
    assert isinstance(data_files, list) or isinstance(data_files, tuple), "case must be either a list or a tuple"
    properties = dict()
    data_itk = [sitk.ReadImage(f) for f in data_files] # [data_itk.shape=(234, 512, 512)]  data_itk[0].shape = (234, 512, 512)

    properties["original_size_of_raw_data"] = np.array(data_itk[0].GetSize())[[2, 1, 0]]
    properties["original_spacing"] = np.array(data_itk[0].GetSpacing())[[2, 1, 0]]
    properties["list_of_data_files"] = data_files
    properties["seg_file"] = seg_file

    properties["itk_origin"] = data_itk[0].GetOrigin()
    properties["itk_spacing"] = data_itk[0].GetSpacing()
    properties["itk_direction"] = data_itk[0].GetDirection()
    # expand dim on axis 0, so that we can use np.vstack() to stack img & seg on axis 0
    data_npy = np.vstack([sitk.GetArrayFromImage(d)[None] for d in data_itk])#   (1, 234, 512, 512)
    if seg_file is not None:
        seg_itk = sitk.ReadImage(seg_file)
        seg_npy = sitk.GetArrayFromImage(seg_itk)[None].astype(np.float32)
    else:
        seg_npy = None
    return data_npy.astype(np.float32), seg_npy, properties

def create_nonzero_mask(data):
    from scipy.ndimage import binary_fill_holes
    # 4个维度
    assert len(data.shape) == 4 , "data must have shape (C, X, Y, Z)"
    nonzero_mask = np.zeros(data.shape[1:], dtype=bool)
    this_mask = data[0] != 0
    nonzero_mask = nonzero_mask | this_mask
    nonzero_mask = binary_fill_holes(nonzero_mask)
    return nonzero_mask

def get_bbox_from_mask(mask, outside_value=0):
    mask_voxel_coords = np.where(mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]

def crop_to_bbox(image, bbox):
    assert len(image.shape) == 3, "only supports 3d images"
    resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
    return image[resizer]

def crop_to_nonzero(data, seg=None, nonzero_label=-1):
    """
    :param data:
    :param seg:
    :param nonzero_label: this will be written into the segmentation map
    :return:
    """
    nonzero_mask = create_nonzero_mask(data)
    bbox = get_bbox_from_mask(nonzero_mask, 0) # [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]

    cropped_data = []
    cropped = crop_to_bbox(data[0], bbox)
    cropped_data.append(cropped[None])
    data = np.vstack(cropped_data)

    if seg is not None:
        cropped_seg = []
        # do same operation for label file
        cropped = crop_to_bbox(seg[0], bbox)
        cropped_seg.append(cropped[None])
        seg = np.vstack(cropped_seg)

    nonzero_mask = crop_to_bbox(nonzero_mask, bbox)[None]
    if seg is not None:
        seg[(seg == 0) & (nonzero_mask == 0)] = nonzero_label
    else:
        nonzero_mask = nonzero_mask.astype(int)
        nonzero_mask[nonzero_mask == 0] = nonzero_label
        nonzero_mask[nonzero_mask > 0] = 0
        seg = nonzero_mask
    return data, seg, bbox

class ImageCropper(object):
    """
        This one finds a mask of nonzero elements (must be nonzero in all modalities) and crops the image to that mask.
        :param num_threads:
        :param output_folder: whete to store the cropped data
        :param list_of_files:
    """
    def __init__(self, num_threads, output_folder=None):
        self.num_threads = num_threads
        self.output_folder = output_folder

    @staticmethod
    def crop(data, properties, seg=None):
        shape_before = data.shape
        data, seg, bbox = crop_to_nonzero(data, seg, nonzero_label=-1)
        shape_after = data.shape
        print("before crop:", shape_before, "after crop:", shape_after, "spacing:",
              np.array(properties["original_spacing"]), "\n")

        properties["crop_bbox"] = bbox
        properties['classes'] = np.unique(seg)
        seg[seg < -1] = 0
        properties["size_after_cropping"] = data[0].shape
        return data, seg, properties

    @staticmethod
    def crop_from_list_of_files(data_files, seg_file=None):
        # data (1, 234, 512, 512)
        # seg (1, 234, 512, 512)
        data, seg, properties = load_case_from_list_of_files(data_files, seg_file)
        return ImageCropper.crop(data, properties, seg)

    def load_crop_save(self, case, case_identifier):
        print(case_identifier)
        data, seg, properties = self.crop_from_list_of_files(case[:-1], case[-1])
        # data.shape=(1, 512, 512, 234)
        # seg.shape=(1, 512, 512, 234)
        all_data = np.vstack((data, seg))
        np.save(join(self.output_folder, "%s.npy" % (case_identifier)), all_data)
        with open(join(self.output_folder, "%s.pkl" %(case_identifier)), 'wb') as f:
            pickle.dump(properties, f)

    def run_cropping(self, list_of_files):
        '''
         interface function, call this to do all process
        :param list_of_files: list of list  like [['C:/Train/imagesTr\\volume-covid19-A-0003_ct.nii.gz',
        'C:/Train/labelsTr\\volume-covid19-A-0003_seg.nii.gz'], ...]
        :return:
        '''

        list_of_args = []
        for j, case in enumerate(list_of_files):
            case_identifier = get_case_identifier(case)
            # case e.g.
            # ['C:/Train/imagesTr\\volume-covid19-A-0003_ct.nii.gz',
            #  'C:/Train/labelsTr\\volume-covid19-A-0003_seg.nii.gz'], ...]
            # case_identifier e.g. volume-covid-A-003
            list_of_args.append((case, case_identifier))

        p = Pool(self.num_threads)
        # how to use p.starmap()?
        # Like `map()`method but the elements of the `iterable`
        # are expected to be iterables as well and will be unpacked as arguments.Hence `func` and (a, b)
        # becomes func(a, b).
        p.starmap(self.load_crop_save, list_of_args)
        p.close()
        p.join()

if __name__ == '__main__':
    imgcropper = ImageCropper(4, './preprocessed/foreground')
    list_of_files = create_lists('c:/Train/imagesTr')
    imgcropper.run_cropping(list_of_files)

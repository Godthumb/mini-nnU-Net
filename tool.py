from file_and_folder_operations import *
import SimpleITK as sitk
import numpy as np


data = load_pickle(r'D:\preprocessed_COVID19\crop_foreground\dataset_properties.pkl')
print(data)

resample_img,resample_seg = np.load(r'D:\preprocessed_COVID19\resample_normalization\volume-covid19-A-0694.npy')
resample_img = sitk.GetImageFromArray(resample_img)
resample_img.SetSpacing(np.array([2.2      , 0.8287265, 0.8287265])[::-1])

resample_seg = sitk.GetImageFromArray(resample_seg)
resample_seg.SetSpacing(np.array([2.2      , 0.8287265, 0.8287265])[::-1])
sitk.WriteImage(resample_img, './0649.nii.gz')
sitk.WriteImage(resample_seg, './0649_seg.nii.gz')

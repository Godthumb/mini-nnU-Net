import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from file_and_folder_operations import subfiles, join
import pickle


class COVID(Dataset):
    def __init__(self, base_dir, ):
        super(COVID, self).__init__()
        self.base_dir = base_dir
        self.resample_data_file_list = subfiles(self.base_dir, None, '.npy', False) # npy files
    
    def __getitem__(self, idx):
        this_case = self.resample_data_file_list[idx]
        case_identifier = self.get_case_identifier(this_case)
        data, seg, properties = self.load_all_data(case_identifier)

        # if foreground pixel exist, use choosed foreground pixel random sample slice
        foreground_classes = np.array(
            [i for i in properties['class_locations'].keys() if len(properties['class_locations'][i]) != 0])
        foreground_classes = foreground_classes[foreground_classes > 0]
        if len(foreground_classes) == 0:
            selected_class = None
            # random_slice = np.random.choice(seg.shape[0])
            print('case does not contain any foreground classes', idx)
            return selected_class
        else:
            selected_class = np.random.choice(foreground_classes)

            voxels_of_that_class = properties['class_locations'][selected_class]
            valid_slices = np.unique(voxels_of_that_class[:, 0])
            random_slice = np.random.choice(valid_slices)
            voxels_of_that_class = voxels_of_that_class[voxels_of_that_class[:, 0] == random_slice]
            voxels_of_that_class = voxels_of_that_class[:, 1:]
        
        
        
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
    

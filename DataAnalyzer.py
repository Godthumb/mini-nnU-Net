import os
import pickle
from multiprocessing import Pool
import numpy as np
join = os.path.join
def write_pickle(obj, file, mode='wb'):
    with open(file, mode) as f:
        pickle.dump(obj, f)

# find file startwith prefix or endwith suffix
# can choose sort, return sorted file list
def subfiles(folder, prefix=None, suffix=None, sort=True):
    l = os.path.join
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isfile(os.path.join(folder, i))
           and (prefix is None or i.startswith(prefix))
           and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res

def get_patient_identifiers_from_cropped_files(folder):
    return [i.split('\\')[-1][:-4] for i in subfiles(folder, suffix='.npy')]

class DatasetAnalyzer(object):
    def __init__(self, folder_with_cropped_data, num_processes=4):
        self.folder_with_cropped_data = folder_with_cropped_data
        # self.patient_identifiers [volume-covid19-A-0003, volume-covid-A-0011, ...]
        self.patient_identifiers = get_patient_identifiers_from_cropped_files(self.folder_with_cropped_data)
        self.num_processes = num_processes
        self.intensityproperties_file = join(self.folder_with_cropped_data, "intensityproperties.pkl")

    def load_properties_of_cropped(self, case_identifier):
        with open(join(self.folder_with_cropped_data, '%s.pkl' %(case_identifier)), 'rb') as f:
            properties = pickle.load(f)
        return properties

    def get_sizes_and_spacings_after_cropping(self):
        sizes = []
        spacings = []
        for c in self.patient_identifiers:
            properties = self.load_properties_of_cropped(c)
            sizes.append(properties['size_after_cropping'])
            spacings.append(properties['original_spacing'])

        return sizes, spacings

    def get_size_reduciton_by_cropping(self):
        size_reduction = dict()
        for p in self.patient_identifiers:
            props = self.load_properties_of_cropped(p)
            shape_before_crop = props["original_size_of_raw_data"]
            shape_after_crop = props["size_after_cropping"]
            size_red = np.prod(shape_after_crop) / np.prod(shape_before_crop)
            size_reduction[p] = size_red

        return size_reduction

    # (2, 296, 512, 512)
    def _get_voxels_in_foreground(self, patient_identifier):
        all_data = np.load(join(self.folder_with_cropped_data, patient_identifier) + '.npy')
        img = all_data[0]
        # [-1, 1, 0]
        foreground_mask = all_data[1] > 0
        voxels = list(img[foreground_mask][::10]) # no need to take every voxel
        return voxels

    @staticmethod
    def _compute_stats(voxels):
        if len(voxels) == 0:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        median = np.median(voxels)
        mean = np.mean(voxels)
        sd = np.std(voxels)
        mn = np.min(voxels)
        mx = np.max(voxels)
        percentile_99_5 = np.percentile(voxels, 99.5)
        percentile_00_5 = np.percentile(voxels, 00.5)
        return median, mean, sd, mn, mx, percentile_99_5, percentile_00_5

    def collect_intensity_properties(self):
        p = Pool(self.num_processes)
        results = dict()
        v = p.map(self._get_voxels_in_foreground, self.patient_identifiers)
        w = []
        # aggregate many list to one list
        for iv in v:
            w += iv
        median, mean, sd, mn, mx, percentile_99_5, percentile_00_5 = self._compute_stats(w)

        # list of list form
        local_props = p.map(self._compute_stats, v)
        props_per_case = dict()

        # aggregate dict of dict
        for i, pat in enumerate(self.patient_identifiers):
            props_per_case[pat] = dict()
            props_per_case[pat]['median'] = local_props[i][0]
            props_per_case[pat]['mean'] = local_props[i][1]
            props_per_case[pat]['sd'] = local_props[i][2]
            props_per_case[pat]['mn'] = local_props[i][3]
            props_per_case[pat]['mx'] = local_props[i][4]
            props_per_case[pat]['percentile_99_5'] = local_props[i][5]
            props_per_case[pat]['percentile_00_5'] = local_props[i][6]

        results['local_props'] = props_per_case
        results['median'] = median
        results['mean'] = mean
        results['sd'] = sd
        results['mn'] = mn
        results['mx'] = mx
        results['percentile_99_5'] = percentile_99_5
        results['percentile_00_5'] = percentile_00_5

        p.close()
        p.join()
        write_pickle(results, self.intensityproperties_file)
        return results

    # class_dict
    # cause we don't make dataset.json, so you should give label to it
    def analyze_dataset(self, class_dict, collect_intensityproperties=True):
        # get all spacings and sizes
        sizes, spacings = self.get_sizes_and_spacings_after_cropping()
        # {"0": "background", "1":"infection"} => [1] only keep foreground
        all_classes = [int(i) for i in class_dict.keys() if int(i) > 0]

        # no need modalities, we only handle CT data here
        # modalities = self.get_modalities()
        if collect_intensityproperties:
            intensityproperties = self.collect_intensity_properties()
        else:
            intensityproperties = None

        # size reduction by cropping
        size_reductions = self.get_size_reduciton_by_cropping()
        dataset_properties = dict()
        dataset_properties['all_sizes'] = sizes
        dataset_properties['all_spacings'] = spacings
        dataset_properties['all_classes'] = all_classes
        # dataset_properties['modalities'] = modalities  # {idx: modality name}
        dataset_properties['intensityproperties'] = intensityproperties
        dataset_properties['size_reductions'] = size_reductions  # {patient_id: size_reduction}

        write_pickle(dataset_properties, join(self.folder_with_cropped_data, "dataset_properties.pkl"))
        return dataset_properties


if __name__ == '__main__':
    # give it class_label & cropped_out_dir
    class_label_dict = {"0": "background", "1": "infection"}
    cropped_out_dir = 'D:/COVID-19-20/preprocessed_COVID19/crop_foreground'
    data_analyzer = DatasetAnalyzer(cropped_out_dir)
    dataset_properties = data_analyzer.analyze_dataset(class_label_dict, collect_intensityproperties=True)








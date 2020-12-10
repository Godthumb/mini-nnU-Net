import os

join = os.path.join
listdir = os.listdir

#
def create_lists(base_img_folder='C:/Train/imagesTr'):
    '''
    :param base_img_folder: folder of img
    :return: list of list  like [['C:/Train/imagesTr\\volume-covid19-A-0003_ct.nii.gz',
    'C:/Train/labelsTr\\volume-covid19-A-0003_seg.nii.gz'], ...]
    '''
    training_files = [join(base_img_folder, img_itk) for img_itk in listdir(base_img_folder)]
    print(training_files)
    lists = []
    for tr in training_files:
        cur_pat = []
        cur_pat.append(tr)
        cur_pat.append(tr.replace('imagesTr', 'labelsTr').replace('ct', 'seg'))
        lists.append(cur_pat)
    return lists

if __name__ == '__main__':
    list_of_lists = create_lists()
    print(list_of_lists)

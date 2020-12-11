import pickle
import os

def load_pickle(file, mode='rb'):
    with open(file, mode) as f:
        a = pickle.load(f)
    return a


def write_pickle(obj, file, mode='wb'):
    with open(file, mode) as f:
        pickle.dump(obj, f)


join = os.path.join
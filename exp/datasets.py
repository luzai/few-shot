import keras
from keras.datasets import cifar10, cifar100
from opts import Config
import numpy as  np


class Dataset:
    def __init__(self, name='cifar10', debug=False):
        if name == 'cifar10':
            self.input_shape = (32, 32, 3)
            self.classes = 10
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        elif name == 'cifar100':
            self.input_shape = (32, 32, 3)
            self.classes = 10
            (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        elif name == 'imagenet':
            raise ValueError('Not Implement')
        y_train = keras.utils.to_categorical(y_train, self.classes)
        y_test = keras.utils.to_categorical(y_test, self.classes)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        self.x_train, self.y_train, self.x_test, self.y_test = x_train, y_train, x_test, y_test
        self.x_test, self.y_test = map(sample_data, [self.x_test, self.y_test])
        if debug:
            self.x_train, self.y_train, self.x_test, self.y_test = map(limit_data,
                                                                       [self.x_train, self.y_train,
                                                                        self.x_test, self.y_test])


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


@static_vars(ind=None)
def sample_data(data, n=256 * 2 + 16):
    if sample_data.ind is None:
        sample_data.ind = np.random.permutation(data.shape[0])[:n]
    return data[sample_data.ind]


def limit_data(data, n=256 + 16):
    return data[:n]


def load_data_svhn():
    import scipy.io as sio
    import commands, os
    import numpy as np
    import os.path as osp
    if not os.path.isdir(osp.join(Config.root_path, 'data/SVHN')):
        os.mkdir(osp.join(Config.root_path, 'data/SVHN'))

    data_set = []
    if not os.path.isfile(osp.join(Config.root_path, 'data/SVHN/train_32x32.mat')):
        data_set.append("train")
    if not os.path.isfile(osp.join(Config.root_path, 'data/SVHN/test_32x32.mat')):
        data_set.append("test")

    try:
        import requests
        from tqdm import tqdm
    except:
        # use pip to install these packages:
        # pip install tqdm
        # pip install requests
        print('please install requests and tqdm package first.')

    for set in data_set:
        print ('download SVHN ' + set + ' data, Please wait.')
        url = "http://ufldl.stanford.edu/housenumbers/" + set + "_32x32.mat"
        response = requests.get(url, stream=True)
        with open("data/SVHN/" + set + "_32x32.mat", "wb") as handle:
            for data in tqdm(response.iter_content()):
                handle.write(data)

    train_data = sio.loadmat(Config.root_path + '/data/SVHN/train_32x32.mat')
    train_x = train_data['X']
    train_y = train_data['y']

    test_data = sio.loadmat(Config.root_path + '/data/SVHN/test_32x32.mat')
    test_x = test_data['X']
    test_y = test_data['y']

    # 1 - 10 to 0 - 9
    train_y = train_y - 1
    test_y = test_y - 1

    train_x = np.transpose(train_x, (3, 0, 1, 2))
    test_x = np.transpose(test_x, (3, 0, 1, 2))

    return (train_x, train_y), (test_x, test_y)

if __name__ == '__main__':
    config = Config(epochs=301, batch_size=256, verbose=2,
                    model_type='vgg5',
                    dataset_type='cifar10',
                    debug=False)

    dataset = Dataset(config.dataset_type, debug=config.debug)

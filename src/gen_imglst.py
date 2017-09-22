'''
For gen /home/wangxinglu/prj/few-shot/data/imglst/img1k.test.txt
'''

import utils
from metadata import *
from utils import *

np.random.seed(64)
train_file = '/home/wangxinglu/prj/few-shot/data/imglst/img1k.train.txt'
test_file = '/home/wangxinglu/prj/few-shot/data/imglst/img1k.test.txt'
prefix = '/home/wangxinglu/prj/few-shot/data/imagenet-raw'
num = 1000


def find_child(tree_, node):
    res = []
    try:
        for node in nx.dfs_preorder_nodes(tree_, node):
            imagepath = get_imagepath(node).strip('.tar')
            if osp.exists(imagepath) and tree_.node[node]['nchild'] == 0:
                res.append(node)
    except Exception as inst:
        print inst, 'wrong'
    return res


leaves = unpickle('nimgs.pkl')
names, nimgs = leaves.keys(), leaves.values()
names, nimgs = cosort(names, nimgs, True)
comb = np.array([names, nimgs]).T
res = np.random.choice(np.arange(comb.shape[0]), size=num, replace=False)
res = np.sort(res)
res = comb[res, :][:, 1]

imgs_train_l, imgs_test_l = [], []

for ind, cls in enumerate(res):
    imgs = glob.glob(prefix + '/' + cls + '*.JPEG')
    imgs = np.array(imgs)
    imgs_train = imgs[:imgs.shape[0] // 9 * 8]
    imgs_test = imgs[imgs.shape[0] // 9 * 8:]

    imgs_train_l.append(
        np.concatenate((imgs_test, np.ones_like(imgs_test) * ind), axis=-1))
    imgs_test_l.append(
        np.concatenate((imgs_train, np.ones_like(imgs_train) * ind), axis=-1)
    )

imgs_train = np.concatenate(imgs_train_l)
np.random.shuffle(imgs_train)

np.savetxt(train_file, imgs_train, delimiter=' ')
np.savetxt(test_file, np.concatenate(imgs_test_l), delimiter=' ')

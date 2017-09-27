'''
For gen /home/wangxinglu/prj/few-shot/data/imglst/img1k.test.txt
'''

import utils
import tensorflow as tf
from metadata import *
from utils import *

# train_file = '/home/wangxinglu/prj/few-shot/data/imglst/img1k.train.txt'
# test_file = '/home/wangxinglu/prj/few-shot/data/imglst/img1k.test.txt'
train_file = '/home/wangxinglu/prj/few-shot/data/imglst/img10k.train.txt'
test_file = '/home/wangxinglu/prj/few-shot/data/imglst/img10k.test.txt'
prefix = '/home/wangxinglu/prj/few-shot/data/imagenet-raw'
num = 10000


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


def cls_sample(num):
    np.random.seed(64)
    # leaves = {}
    # for node in tf.gfile.ListDirectory(prefix):
    #     leaves[node] = len(tf.gfile.ListDirectory(prefix + '/' + node))

    # pickle(leaves,'nimgs.pkl')
    leaves = unpickle('nimgs.pkl')

    len(leaves)
    names, nimgs = leaves.keys(), leaves.values()
    names, nimgs = cosort(names, nimgs, True)
    names = names[nimgs >= 10]
    nimgs = nimgs[nimgs >= 10]
    comb = np.array([names, nimgs]).T

    res = np.random.choice(np.arange(comb.shape[0]), size=num, replace=False)
    res = np.sort(res)
    res_nimgs = comb[res, :][:, 1]
    res = comb[res, :][:, 0]
    return res, res_nimgs

@chdir_to_root
def gen_imglst(names):
    os.chdir(prefix)

    imgs_train_l, imgs_test_l = [], []

    for ind, cls in enumerate(names):
        imgs = tf.gfile.Glob(cls + '/*.JPEG')
        imgs = np.array(imgs)
        imgs_train = imgs[:imgs.shape[0] * 9 // 10]
        imgs_test = imgs[imgs.shape[0] * 9 // 10:]

        imgs_train_l.append(
            np.stack((imgs_train, np.ones_like(imgs_train, dtype=int) * ind), axis=-1)
        )
        imgs_test_l.append(
            np.stack((imgs_test, np.ones_like(imgs_test, dtype=int) * ind), axis=-1)
        )
        # if ind > 100:
        #     break

    imgs_train = np.concatenate(imgs_train_l, axis=0)
    np.random.shuffle(imgs_train)

    np.savetxt(train_file, imgs_train, delimiter=' ', fmt='%s')
    np.savetxt(test_file, np.concatenate(imgs_test_l, axis=0), delimiter=' ', fmt='%s')

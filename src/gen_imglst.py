'''
For gen /home/wangxinglu/prj/few-shot/data/imglst/img1k.test.txt
'''

import utils
import tensorflow as tf
from metadata import *
from utils import *

# train_file = '/home/wangxinglu/prj/few-shot/data/imglst/img1k.train.txt'
# test_file = '/home/wangxinglu/prj/few-shot/data/imglst/img1k.test.txt'

train_file = '/home/wangxinglu/prj/few-shot/data/imglst/img10k.train.disk.txt'
test_file = '/home/wangxinglu/prj/few-shot/data/imglst/img10k.test.disk.txt'
prefix = '/home/wangxinglu/prj/few-shot/data/imagenet-raw'

train_file2 = '/home/wangxinglu/prj/few-shot/data/imglst/img10k.train.redis.txt'
test_file2 = '/home/wangxinglu/prj/few-shot/data/imglst/img10k.test.redis.txt'
prefix2 = '/mnt/nfs1703/kchen/imagenet-raw-trans-to-redis'

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
    base_p = 1. / comb.shape[0]

    p = np.concatenate((np.ones(11803) * 1.35 * base_p, np.ones(1835) * 0.7 * base_p, np.ones(717) * 1.35 * base_p))
    p = p / p.sum()

    res = np.random.choice(np.arange(comb.shape[0]), size=num, replace=False, p=p)

    res = np.sort(res)
    res_nimgs = comb[res, :][:, 1]
    res = comb[res, :][:, 0]
    return res, res_nimgs


@chdir_to_root
def gen_imglst(names, prefix, train_file, test_file):
    os.chdir(prefix)
    imgs_train_l, imgs_test_l = [], []
    for ind, cls in enumerate(names):
        if not osp.exists(cls): continue
        imgs = tf.gfile.Glob(cls + '/*.JPEG')

        imgs = np.array(imgs)
        imgs_test = np.random.choice(imgs, max(3, imgs.shape[0] * 1 // 10), replace=False)
        imgs_train = np.setdiff1d(imgs, imgs_test)
        # imgs_train = imgs[:imgs.shape[0] * 9 // 10]
        # imgs_test = imgs[imgs.shape[0] * 9 // 10:]
        # imgs_test.shape, imgs_train.shape, imgs.shape
        imgs_train_l.append(
            np.stack((imgs_train, np.ones_like(imgs_train, dtype=int) * ind), axis=-1)
        )
        imgs_test_l.append(
            np.stack((imgs_test, np.ones_like(imgs_test, dtype=int) * ind), axis=-1)
        )

    imgs_train = np.concatenate(imgs_train_l, axis=0)
    np.random.shuffle(imgs_train)

    np.savetxt(train_file, imgs_train, delimiter=' ', fmt='%s')
    np.savetxt(test_file, np.concatenate(imgs_test_l, axis=0), delimiter=' ', fmt='%s')


names, nimgs = cls_sample(num)
gen_imglst(names, prefix, train_file, test_file)
gen_imglst(names, prefix2, train_file2, test_file2)

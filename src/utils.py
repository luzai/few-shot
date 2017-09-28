# from __future__ import absolute_import

import os, csv, time, cPickle, \
    random, os.path as osp, \
    subprocess, json, matplotlib, \
    numpy as np, pandas as pd, \
    glob, re, networkx as nx, \
    h5py, yaml, copy, multiprocessing as mp, \
    pandas as pd

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from IPython import embed
from IPython.display import display, HTML, SVG

root_path = osp.normpath(
    osp.join(osp.abspath(osp.dirname(__file__)), "..")
)


def append_file(line, file=None):
    file = file or 'append.txt'
    with open(file, 'a') as  f:
        f.writelines(line + '\n')


def cpu_priority(level=19):
    import psutil
    p = psutil.Process(os.getpid())
    p.nice(level)


def to_int(x):
    return np.around(x).astype(int)


def init_dev(n=(0,)):
    from os.path import expanduser
    home = expanduser("~")
    if isinstance(n, int):
        n = (n,)
    devs = ''
    for n_ in n:
        devs += str(n_) + ','
    os.environ["CUDA_VISIBLE_DEVICES"] = devs
    os.environ['PATH'] = home + '/cuda-8.0/bin:' + os.environ['PATH']
    os.environ['PATH'] = home + 'anaconda2/bin:' + os.environ['PATH']
    os.environ['PATH'] = home + '/usr/local/cuda-8.0/bin:' + os.environ['PATH']

    os.environ['LD_LIBRARY_PATH'] = home + '/cuda-8.0/lib64'
    os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-8.0/lib64'
    # os.environ['PYTHONWARNINGS'] = "ignore"


def allow_growth_config():
    import tensorflow as tf
    _sess_config = tf.ConfigProto(allow_soft_placement=True)
    _sess_config.gpu_options.allow_growth = True
    return _sess_config


def allow_growth():
    import tensorflow as tf
    tf_graph = tf.get_default_graph()
    _sess_config = tf.ConfigProto(allow_soft_placement=True)
    _sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=_sess_config, graph=tf_graph)
    import keras.backend as K
    K.set_session(sess)
    return sess


def get_dev(n=1, ok=(0, 1, 2, 3, 4, 5, 6, 7)):
    import GPUtil, time
    print('Auto select gpu')
    GPUtil.showUtilization()

    def _limit(devs, ok):
        return [dev for dev in devs if dev in ok]

    devs = GPUtil.getAvailable(order='memory', maxLoad=0.5, maxMemory=0.5, limit=n)  #
    devs = _limit(devs, ok)
    if len(devs) >= 1:
        print('available {}'.format(devs))
        # GPUtil.showUtilization()
        return devs[0] if n == 1 else devs
    while len(devs) == 0:
        devs = GPUtil.getAvailable(order='random', maxLoad=0.98, maxMemory=0.98, limit=n)
        devs = _limit(devs, ok)
        if len(devs) >= 1:
            print('available {}'.format(devs))
            GPUtil.showUtilization()
            return devs[0] if n == 1 else devs
        print('no device avelaible')
        GPUtil.showUtilization()
        time.sleep(60)  # 60 * 3


def grid_iter(tmp):
    res = cartesian(tmp.values())
    np.random.shuffle(res)
    for res_ in res:
        yield dict(zip(tmp.keys(), res_))


def shuffle_iter(iter):
    iter = list(iter)
    np.random.shuffle(iter)
    for iter_ in iter:
        yield iter_


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """
    if len(arrays) == 0:
        return []
    arrays = [np.asarray(x) for x in arrays]
    # dtype = arrays[0].dtype
    dtype = object
    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in xrange(1, arrays[0].size):
            out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]
    return out


def optional_arg_decorator(fn):
    def wrapped_decorator(*args):
        if len(args) == 1 and callable(args[0]):
            return fn(args[0])
        else:
            def real_decorator(decoratee):
                return fn(decoratee, *args)

            return real_decorator

    return wrapped_decorator


def randomword(length):
    import random, string
    return ''.join(random.choice(string.lowercase) for i in range(length))


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


def dict2df(my_dict):
    tensor_d = {}
    for k, v in my_dict.iteritems():
        #     print k,v.shape
        if k[1] not in tensor_d:
            tensor_d[k[1]] = pd.Series(name=k[1], index=pd.Int64Index([]))
        tensor_d[k[1]][k[0]] = v
    return pd.DataFrame.from_dict(tensor_d)


def cosort(tensor, y, return_y=False):
    comb = zip(tensor, y)
    comb_sorted = sorted(comb, key=lambda x: x[1])
    if not return_y:
        return np.array([comb_[0] for comb_ in comb_sorted])
    else:
        return np.array([comb_[0] for comb_ in comb_sorted]), np.array([comb_[1] for comb_ in
                                                                        comb_sorted])


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.start_time = time.time()
        # logger.info('time pass {}'.format(self.diff))
        return self.diff


timer = Timer()


@optional_arg_decorator
def timeit(fn, info=''):
    def wrapped_fn(*arg, **kwargs):
        timer = Timer()
        timer.tic()
        res = fn(*arg, **kwargs)
        diff = timer.toc()
        print((info + 'takes time {}').format(diff))
        return res

    return wrapped_fn


def read_json(file_path):
    with open(file_path, 'r') as handle:
        fixed_json = ''.join(line for line in handle if not '//' in line)
        employee_data = json.loads(fixed_json)
    return employee_data


def write_json(obj, file_path):
    dir_name = osp.dirname(file_path)
    if dir_name != '':
        mkdir_p(dir_name, delete=False)
    with open(file_path, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def pickle(data, file_path):
    mkdir_p(osp.dirname(file_path), delete=False)
    print 'pickle into', file_path
    with open(file_path, 'wb') as f:
        cPickle.dump(data, f, cPickle.HIGHEST_PROTOCOL)


def unpickle(file_path):
    with open(file_path, 'rb') as f:
        data = cPickle.load(f)
    return data


def write_df(df, path):
    df.to_hdf(path, 'df', mode='w')


def read_df(path):
    return pd.read_hdf(path, 'df')


def mkdir_p(path, delete=False):
    if path == '': return
    print('mkdir -p  ' + path)
    if delete:
        rm(path)
    if not osp.exists(path):
        subprocess.call(('mkdir -p ' + path).split())


def shell(cmd, block=True):
    import os
    my_env = os.environ.copy()
    home = os.path.expanduser('~')
    my_env["PATH"] = home + "/anaconda2/bin:" + my_env["PATH"]
    if block:
        # subprocess.call(cmd.split())
        task = subprocess.Popen(cmd,
                                shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                env=my_env)
        return task.communicate()
    else:
        print 'Non-block!'
        task = subprocess.Popen(cmd,
                                shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                env=my_env)
        return task


def ln(path, to_path):
    if osp.exists(to_path):
        # pass
        print 'error! exist ' + to_path
    path = osp.abspath(path)
    cmd = "ln -s " + path + " " + to_path
    print cmd
    proc = subprocess.Popen(cmd, shell=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    return proc


def tar(path, to_path=None):
    if not osp.exists(path):
        return
    if not osp.exists(to_path):
        mkdir_p(to_path)
    if os.path.exists(to_path) and not len(os.listdir(to_path)) == 0:
        rm(path)
        return
    if to_path is not None:
        cmd = "tar xf " + path + " -C " + to_path
        print cmd
    else:
        cmd = "tar xf " + path
    shell(cmd, block=True)
    if os.path.exists(path):
        rm(path)


def rmdir(path):
    cmd = "rmdir " + path
    shell(cmd)


def rm(path, block=True):
    print('Are you sure to rm {}'.format(path))
    cmd = 'rm -rf ' + path
    return shell(cmd, block=block)


def show_img(path):
    from IPython.display import Image

    fig = Image(filename=(path))
    return fig


def show_pdf(path):
    from IPython.display import IFrame
    path = osp.relpath(path)
    return IFrame(path, width=600, height=300)


def print_graph_info():
    import tensorflow as tf
    graph = tf.get_default_graph()
    graph.get_tensor_by_name("Placeholder:0")
    layers = [op.name for op in graph.get_operations() if op.type == "Placeholder"]
    print [graph.get_tensor_by_name(layer + ":0") for layer in layers]
    print [op.type for op in graph.get_operations()]
    print [n.name for n in tf.get_default_graph().as_graph_def().node]
    print [v.name for v in tf.global_variables()]
    print graph.get_operations()[20]


def chdir_to_root(fn):
    def wrapped_fn(*args, **kwargs):
        restore_path = os.getcwd()
        os.chdir(root_path)
        res = fn(*args, **kwargs)
        os.chdir(restore_path)
        return res

    return wrapped_fn


def scp(src, dest, dry_run=False):
    cmd = ('scp -r ' + src + ' ' + dest)
    print cmd
    if dry_run: return
    return shell(cmd, block=False)


def read_list(file):
    if osp.exists(file):
        lines = np.genfromtxt(file, dtype='str')
        return lines
    else:
        return []


@chdir_to_root
def vis_nx(graph, name='default', show=False):
    import networkx as nx
    path = osp.dirname(name)
    name = osp.basename(name)
    if path == '':
        path = name
    mkdir_p(osp.join(root_path, "output", path), delete=False)
    try:
        plt.close('all')
        nx.draw(graph, with_labels=True)
        if show:
            plt.show()
        plt.savefig('graph.png')
        print ' nx plot success', path
    except Exception as inst:
        print 'error', inst


@chdir_to_root
def to_single_dir(dir='tfevents'):
    for parent, dirnames, filenames in os.walk(dir):
        filenames = sorted(filenames)
        if len(filenames) == 1:
            continue
        for ind, fn in enumerate(filenames):
            mkdir_p(parent + '/' + str(ind), False)
            mv(parent + '/' + fn, parent + '/' + str(ind) + '/')
        print parent, filenames


def cp(from_path, to):
    subprocess.call(('cp -r ' + from_path + ' ' + to).split())


def mv(from_path, to):
    if not osp.exists(to):
        mkdir_p(to)
    if not isinstance(from_path, list):
        subprocess.call(('mv ' + from_path + ' ' + to).split())
    else:
        for from_ in from_path:
            subprocess.call(('mv ' + from_ + ' ' + to).split())


def dict_concat(d_l):
    d1 = d_l[0].copy()
    for d in d_l[1:]:
        d1.update(d)
    return d1


@chdir_to_root
def merge_dir(dir_l):
    for dir in dir_l:
        for parent, dirnames, filenames in os.walk(dir):
            if len(filenames) != 1:
                continue
            mkdir_p('/'.join(['_res'] + parent.split('/')[1:]), delete=False)
            if not osp.exists('/'.join(['_res'] + parent.split('/')[1:]) + '/' + filenames[0]):
                cp(parent + '/' + filenames[0], '/'.join(['_res'] + parent.split('/')[1:]))
            else:
                print parent


def clean_name(name):
    import re
    name = re.findall('([a-zA-Z0-9/-]+)(?::\d+)?', name)[0]
    name = re.findall('([a-zA-Z0-9/-]+)(?:_\d+)?', name)[0]
    return name


def dict2str(others):
    name = ''
    for key, val in others.iteritems():
        name += '_' + str(key)
        if isinstance(val, dict):
            name += '_' + dict2str(val)
        elif isinstance(val, list):
            for val_ in val:
                name += '-' + str(val_)
        else:
            name += '_' + str(val)
    return name


def list2str(li, delimier=''):
    name = ''
    for name_ in li:
        name += (str(name_) + delimier)

    return name


def write_list(file, l, sort=True):
    l = np.array(l)
    if sort:
        l = np.sort(l, axis=0)
    np.savetxt(file, l, delimiter=' ')


def rsync(from_, to):
    cmd = ('rsync -avzP ' + from_ + ' ' + to)
    print cmd
    return shell(cmd, block=False)


# @chdir_to_root
def tar_imagenet():
    os.chdir(root_path)
    os.chdir('data/imagenet22k-raw')
    files = glob.glob('*.tar')
    # task_l = []
    for file in shuffle_iter(files):
        if file not in glob.glob('*.tar'): continue
        mkdir_p(file.strip('.tar'), delete=True)
        tar(file, file.strip('.tar'))
        rm(file, block=True)
        while os.path.exists(file): pass
    print 'ok'


@chdir_to_root
def gen_imagenet22k_label():
    all = [node for node in os.listdir('./data/images/') if '.tar' not in node]
    os.chdir('data')
    write_list('imagenet22k.txt', all)
    return np.sort(all)


def transfer():
    images = 'images'
    os.chdir('/DATA/luzai/imagenet-python')
    task_l = []
    for file in shuffle_iter(os.listdir(images)):
        # task_l.append(rsync(images + '/' + file, 'mm:/mnt/nfs1703/kchen/imagenet10k-raw/'))
        if file not in os.listdir('./tmp/'):
            task_l.append(rsync(images + '/' + file, './tmp/'))
        else:
            rm(images + '/' + file, block=False)
        if len(task_l) >= 20:
            [task.communicate() for task in task_l]
            task_l = []


@chdir_to_root
def check_jpeg():
    for img in glob.iglob('data/imagenet22k-raw/*/*.JPEG'):
        try:
            im = plt.imread(img)
            print img
        except Exception as inst:
            print img, inst
            # from IPython import embed; embed()


if __name__ == '__main__':
    exit(1)

    import multiprocessing as mp

    result = []
    pool = mp.Pool(64)
    result.append(pool.apply_async(func=tar_imagenet, args=()))
    pool.close()
    pool.join()
    for res in result:
        print res.get()
    # gen_imagenet22k_label()
    # check_jpeg()
    pass

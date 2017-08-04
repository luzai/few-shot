import os, csv, time, cPickle, \
    random, os.path as osp, \
    subprocess, json, matplotlib, \
    numpy as np, GPUtil, pandas as pd, \
    glob, re

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from IPython import embed
from IPython.display import display, HTML, SVG
from logs import logger

root_path = osp.normpath(
    osp.join(osp.abspath(osp.dirname(__file__)), "..")
)


def init_dev(n=0):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(n)
    os.environ['PATH'] = '/home/gyzhang/cuda-8.0/bin:' + os.environ['PATH']
    os.environ['PATH'] = '/home/wangxinglu/anaconda2/bin:' + os.environ['PATH']
    os.environ['PATH'] = '/usr/local/cuda-8.0/bin:' + os.environ['PATH']

    os.environ['LD_LIBRARY_PATH'] = '/home/gyzhang/cuda-8.0/lib64'
    os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-8.0/lib64'
    # os.environ['PYTHONWARNINGS'] = "ignore"


def allow_growth():
    import tensorflow as tf
    tf_graph = tf.get_default_graph()
    _sess_config = tf.ConfigProto(allow_soft_placement=True)
    _sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=_sess_config, graph=tf_graph)
    import keras.backend as K
    K.set_session(sess)


def get_dev(n=1, ok=(0, 1, 2, 3)):
    import GPUtil, time
    logger.info('Auto select gpu')
    GPUtil.showUtilization()

    def _limit(devs, ok):
        return [dev for dev in devs if dev in ok]

    devs = GPUtil.getAvailable(order='memory', maxLoad=0.5, maxMemory=0.5, limit=n)
    devs = _limit(devs, ok)
    if len(devs) >= 1:
        logger.info('available {}'.format(devs))
        GPUtil.showUtilization()
        return devs[0] if n == 1 else devs
    while len(devs) == 0:
        devs = GPUtil.getAvailable(order='random', maxLoad=0.98, maxMemory=0.5, limit=n)
        devs = _limit(devs, ok)
        if len(devs) >= 1:
            logger.info('available {}'.format(devs))
            GPUtil.showUtilization()
            return devs[0] if n == 1 else devs
        logger.info('no device avelaible')
        GPUtil.showUtilization()
        time.sleep(5)  # 60 * 3


def optional_arg_decorator(fn):
    def wrapped_decorator(*args):
        if len(args) == 1 and callable(args[0]):
            return fn(args[0])

        else:
            def real_decorator(decoratee):
                return fn(decoratee, *args)

            return real_decorator

    return wrapped_decorator


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


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
        timer.tic()
        res = fn(*arg, **kwargs)
        diff = timer.toc()
        logger.info((info + 'takes time {}').format(diff))
        return res

    return wrapped_fn


def read_json(file_path):
    with open(file_path, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, file_path):
    mkdir_p(osp.dirname(file_path), delete=False)
    with open(file_path, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def pickle(data, file_path):
    mkdir_p(osp.dirname(file_path), delete=False)
    with open(file_path, 'wb') as f:
        cPickle.dump(data, f, cPickle.HIGHEST_PROTOCOL)


def unpickle(file_path):
    with open(file_path, 'rb') as f:
        data = cPickle.load(f)
    return data


def mkdir_p(path, delete=False):
    if delete:
        rm(path)
    if not osp.exists(path):
        subprocess.call(('mkdir -p ' + path).split())


def rm(path):
    subprocess.call(('rm -rf ' + path).split())


def i_vis_model(model):
    from keras.utils import vis_utils
    return SVG(vis_utils.model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))


def vis_model(model, name='model', show_shapes=True):
    import keras
    from keras.utils import vis_utils
    path = osp.dirname(name)
    name = osp.basename(name)
    if path == '':
        path = name
    sav_path = osp.join(Config.root_path, "output", path)
    mkdir_p(sav_path, delete=False)
    keras.models.save_model(model, osp.join(sav_path, name + '.h5'))
    try:
        # vis_utils.plot_model(model, to_file=osp.join(sav_path, name + '.pdf'), show_shapes=show_shapes)
        vis_utils.plot_model(model, to_file=osp.join(sav_path, name + '.png'), show_shapes=show_shapes)
    except Exception as inst:
        logger.error("cannot keras.plot_model {}".format(inst))


def vis_graph(graph, name='net2net', show=False):
    import networkx as nx
    path = osp.dirname(name)
    name = osp.basename(name)
    if path == '':
        path = name
    mkdir_p(osp.join(root_path, "output", path), delete=False)
    restore_path = os.getcwd()
    os.chdir(osp.join(root_path, "output", path))
    with open(name + "_graph.json", "w") as f:
        f.write(graph.to_json())
    try:
        plt.close('all')
        nx.draw(graph, with_labels=True)
        if show:
            plt.show()
        plt.savefig('graph.png')
        # plt.close('all')
    except Exception as inst:
        logger.warning(inst)
    os.chdir(restore_path)


def count_weight(model):
    import keras.backend as K
    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)]) * 4. / 1024. / 1024.
    # convert to MB
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]) * 4. / 1024. / 1024.

    return trainable_count, non_trainable_count


def print_graph_info():
    import tensorflow as tf
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    graph = tf.get_default_graph()
    graph.get_tensor_by_name("Placeholder:0")
    layers = [op.name for op in graph.get_operations() if op.type == "Placeholder"]
    print [graph.get_tensor_by_name(layer + ":0") for layer in layers]
    print [op.type for op in graph.get_operations()]
    print [n.name for n in tf.get_default_graph().as_graph_def().node]
    print [v.name for v in tf.global_variables()]
    print graph.get_operations()[20]


def i_vis_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    import tensorflow as tf
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph' + str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:800px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))


def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    import tensorflow as tf
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add()
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = tf.compat.as_bytes("<stripped %d bytes>" % size)
    return strip_def


def add_indent(str):
    import re
    return re.sub('\n', '\n\t\t', str)


def chdir_to_root(fn):
    def wrapped_fn(*args, **kwargs):
        restore_path = os.getcwd()
        os.chdir(Config.root_path)
        res = fn(*args, **kwargs)
        os.chdir(restore_path)
        return res

    return wrapped_fn


@chdir_to_root
def to_single_dir(dir='tfevents'):
    for parent, dirnames, filenames in os.walk(dir):
        filenames = sorted(filenames)
        if len(filenames) == 1:
            continue
        for ind, fn in enumerate(filenames):
            mkdir_p(parent + '/' + str(ind), False)
            move(parent + '/' + fn, parent + '/' + str(ind) + '/')
        print parent, filenames


def copy(from_path, to):
    subprocess.call(('cp ' + from_path + ' ' + to).split())


def move(from_path, to):
    subprocess.call(('mv ' + from_path + ' ' + to).split())


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
                copy(parent + '/' + filenames[0], '/'.join(['_res'] + parent.split('/')[1:]))
            else:
                print parent


@chdir_to_root
def parse_dir_name(dir='_res'):
    dir_l = glob.glob(dir + '/*/*')
    res = []
    for path in dir_l:
        name = path.split('/')[1]
        name_l = name.split('_')
        name_l[-1] = '{:.2e}'.format(float(name_l[-1]))
        res.append(name_l)
        _name = '_'.join(name_l)
        # move( '/'.join(path.split('/')[:2]), '/'.join(path.split('/')[:1]+ [_name]))
    return res

def clean_name(name):
    import re
    name = re.findall('([a-zA-Z0-9/]+)(?::\d+)?', name)[0]
    name = re.findall('([a-zA-Z0-9/]+)(?:_\d+)?', name)[0]
    return name

if __name__ == '__main__':
    # to_single_dir()
    # merge_dir(['loss', 'tfevents'])
    # print np.array( parse_dir_name())
    # print np.array(parse_dir_name('tfevents_loss'))
    print root_path

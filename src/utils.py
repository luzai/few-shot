import os, csv, time, cPickle, \
  random, os.path as osp, \
  subprocess, json, matplotlib, \
  numpy as np, GPUtil, pandas as pd, \
  glob, re

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from IPython import embed
from IPython.display import display, HTML, SVG

root_path = osp.normpath(
    osp.join(osp.abspath(osp.dirname(__file__)), "..")
)


def init_dev(n=0):
  import os
  from os.path import expanduser
  home = expanduser("~")
  os.environ["CUDA_VISIBLE_DEVICES"] = str(n)
  os.environ['PATH'] = home + '/cuda-8.0/bin:' + os.environ['PATH']
  os.environ['PATH'] = home + 'anaconda2/bin:' + os.environ['PATH']
  os.environ['PATH'] = home + '/usr/local/cuda-8.0/bin:' + os.environ['PATH']
  
  os.environ['LD_LIBRARY_PATH'] = home + '/cuda-8.0/lib64'
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
  from logs import logger
  import GPUtil, time
  print('Auto select gpu')
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
    print('no device avelaible')
    GPUtil.showUtilization()
    time.sleep(60)  # 60 * 3


def grid_iter(tmp):
  res = cartesian(tmp.values())
  np.random.shuffle(res)
  for res_ in res:
    yield dict(zip(tmp.keys(), res_))


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
    from logs import logger
    timer = Timer()
    timer.tic()
    res = fn(*arg, **kwargs)
    diff = timer.toc()
    logger.debug((info + 'takes time {}').format(diff))
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
  assert path != ''
  from logs import logger
  logger.info('mkdir -p  '+ path)
  if delete:
    rm(path)
  if not osp.exists(path):
    subprocess.call(('mkdir -p ' + path).split())


def rm(path):
  subprocess.call(('rm -rf ' + path).split())

def show_img(path):
  from IPython.display import Image
  
  fig = Image(filename=(path))
  return fig

def show_pdf(path):
  from IPython.display import IFrame
  path=osp.relpath(path)
  return IFrame(path, width=600, height=300)

def i_vis_model(model):
  from keras.utils import vis_utils
  return SVG(vis_utils.model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))


def vis_model(model, name='model', show_shapes=True):
  import keras
  from logs import logger
  from keras.utils import vis_utils
  path = osp.dirname(name)
  name = osp.basename(name)
  if path == '':
    path = name
  sav_path = osp.join(root_path, "output", path)
  mkdir_p(sav_path, delete=False)
  keras.models.save_model(model, osp.join(sav_path, name + '.h5'))
  try:
    # vis_utils.plot_model(model, to_file=osp.join(sav_path, name + '.pdf'), show_shapes=show_shapes)
    vis_utils.plot_model(model, to_file=osp.join(sav_path, name + '.png'), show_shapes=show_shapes)
  except Exception as inst:
    logger.error("cannot keras.plot_model {}".format(inst))


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
    os.chdir(root_path)
    res = fn(*args, **kwargs)
    os.chdir(restore_path)
    return res
  
  return wrapped_fn


@chdir_to_root
def vis_graph(graph, name='net2net', show=False):
  from logs import logger
  import networkx as nx
  path = osp.dirname(name)
  name = osp.basename(name)
  if path == '':
    path = name
  mkdir_p(osp.join(root_path, "output", path), delete=False)
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


@chdir_to_root
def get_config(key):
  import sys
  sys.path.append(root_path)
  from hypers import hyper
  sys.path.pop()
  if key in hyper:
    return hyper[key]
  else:
    return hyper[hyper['use']][key]


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

def check_md5sum():
  for parant_folder in ['stat301', 'stat101', 'stat101_10', 'stat301_10']:
    path = '../output' + parant_folder.strip('stat') + '_all_stat'
    # '_all_stat/resnet10_cifar100_lr_1.00e-02'
    # print glob.glob(path+'/*')
    tpath = glob.glob(path + '/*')[0]
    # print tpath
    
    
    path = '../' + parant_folder + '/resnet10_cifar100_lr_1.00e-02/mi*/c*'
    file = glob.glob(path)[0]
    
    subprocess.call(('md5sum ' + file).split())


def merge_pdf(names):
  from pyPdf import PdfFileWriter, PdfFileReader
  from logs import logger
  
  # Creating a routine that appends files to the output file
  def append_pdf(input, output):
    [output.addPage(input.getPage(page_num)) for page_num in range(input.numPages)]
  
  # Creating an object where pdf pages are appended to
  output = PdfFileWriter()
  
  # Appending two pdf-pages from two different files
  for name in names:
    if not osp.exists(name):
      logger.warning(name + 'do not exist')
    append_pdf(PdfFileReader(open(name, "rb")), output)
  
  # Writing all the collected pages to a f  ile
  output.write(open(
      (osp.dirname(names[0]) + "/merged.pdf").rstrip('/')
      , "wb"))


if __name__ == '__main__':
  # to_single_dir()
  # merge_dir(['loss', 'tfevents'])
  # print np.array( parse_dir_name())
  # print np.array(parse_dir_name('tfevents_loss'))
  
  pass

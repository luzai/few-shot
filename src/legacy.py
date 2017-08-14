from logs import logger
from configs import Config
from datasets import Dataset
import time, utils, glob, os, re, copy
import numpy as np, os.path as osp, pandas as pd, matplotlib.pylab as plt


def get_shape(arr):
  return np.array(arr).shape

def df2arr(df):
  df = df.copy()
  df = df.unstack()
  shape = map(len, df.index.levels)
  arr = np.full(shape, np.nan)
  arr[df.index.labels] = df.values.flat
  return arr, MultiIndexFacilitate(df.index)


def arr2df(arr, indexf):
  df2 = pd.DataFrame(arr.flatten(), index=indexf.index)
  df2.reset_index().pivot_table(values=0, index=indexf.names[-1:], columns=indexf.names[:-1])
  return df2

# def select(arr,indexf,name,levels):
#   inds = indexf.level2inds(levels)


def t_sne(visualizer, model_type, dataset_type, start_lr):
  logger.info('start ' + '_'.join((model_type, dataset_type, 'start_lr', str(start_lr))))
  stat_df = visualizer.stat_df.copy()
  stat_df = visualizer.select(stat_df, 'lr', '1.00e[+-]0[{}-9].*'.format(start_lr))
  stat_df = visualizer.select(stat_df, 'name', '(?:.*/dense/.*|.*/conv2d/.*)')
  stat_df = visualizer.select(stat_df, 'dataset_type', dataset_type, regexp=False)
  stat_df = visualizer.select(stat_df, 'model_type', model_type, regexp=False)
  
  stat_df, suptitle = drop_level(stat_df, keep_num_levels=2)
  
  perf_df = visualizer.perf_df.copy()
  perf_df = visualizer.select(perf_df, 'lr', '1.00e[+-]0[{}-9].*'.format(start_lr))
  perf_df = visualizer.select(perf_df, 'dataset_type', dataset_type, regexp=False)
  perf_df = visualizer.select(perf_df, 'model_type', model_type, regexp=False)
  perf_df = visualizer.select(perf_df, 'name', 'val_acc', regexp=False)
  
  perf_df, suptitle = drop_level(perf_df, keep_num_levels=2)
  
  stats = []
  colors = []
  val_accs = []
  
  levels, names, name2level, name2ind = get_columns_alias(stat_df.columns)
  for ind, lr in enumerate(name2level['lr']):
    _perf_df = visualizer.select(perf_df, 'lr', lr, regexp=False)
    _perf_df, _suptitle = drop_level(_perf_df, other_name=['lr'],
                                     keep_num_levels=1)
    logger.debug('{}'.format(_perf_df.columns))
    val_accs.append(_perf_df.as_matrix())
    
    _stat_df = visualizer.select(stat_df, 'lr', lr, regexp=False)
    _stat_df, _suptitle = drop_level(_stat_df, other_name=['lr'],
                                     keep_num_levels=1)
    colors.append(ind)
    stats.append(_stat_df.as_matrix())
  
  stats_all = np.concatenate(stats, axis=0)
  # colors = np.concatenate(colors, axis=0)
  # val_accs = np.concatenate(val_accs, axis=0)
  
  stats_all = preprocessing.scale(stats_all, axis=0)
  # for multitime in range(1):
  stats_dim2_all = manifold.TSNE(n_components=2).fit_transform(stats_all)
  
  stats_dim2 = []
  ind = 0
  for stat in stats:
    stats_dim2.append(stats_dim2_all[ind:ind + stat.shape[0]])
    ind += stat.shape[0]
  
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  if globals()['dbg']:
    dur = 0.1
  else:
    dur = 9.
  freq = stats_dim2[0].shape[0] / dur
  
  def make_frame_mpl(t):
    ax.clear()
    time_i = int(t * freq)
    for ind, (stat_dim2, color, val_acc) in enumerate(zip(stats_dim2, colors, val_accs)):
      val_acc[:time_i].reshape((time_i,))
      lr = name2level['lr'][ind]
      ax.scatter(stat_dim2[:time_i, 0],
                 stat_dim2[:time_i, 1],
                 val_acc[:time_i].reshape((time_i,)),
                 c=get_colors(lr),
                 label=lr
                 # cmap=plt.cm.Spectral,
                 )
    
    ax.legend(loc='upper right')
    # ax.get_xaxis().set_ticklabels([])
    # ax.get_yaxis().set_ticklabels([])
    ax.set_xlim3d([stats_dim2_all[:, 0].min(), stats_dim2_all[:, 0].max()])
    ax.set_ylim3d([stats_dim2_all[:, 1].min(), stats_dim2_all[:, 1].max()])
    ax.set_zlim3d([0, 1])
    ax.view_init(elev=20., azim=t / dur * 130.)
    fig.suptitle('_'.join((model_type, dataset_type, 'start_lr', str(start_lr))))
    return mplfig_to_npimage(fig)
    
    ## animation
    # animation = mpy.VideoClip(make_frame_mpl, duration=dur)
    # animation.write_gif('_'.join((model_type, dataset_type, 'start_lr', str(start_lr))) + '.gif', fps=20)
    ## plot fig
    # make_frame_mpl(dur)
    # ax.view_init(elev=90., azim=0.)
    # # ax.view_init(elev=20., azim=45.)
    # plt.savefig(parant_folder + '_'.join((model_type, dataset_type, 'start_lr', str(start_lr))) + '.pdf')


def subplots(visualizer, path_suffix):
  perf_df = visualizer.perf_df.copy()
  # plot only val acc
  # perf_df = select(perf_df, {'name': 'val_acc$'})
  perf_df.columns = merge_level(perf_df.columns, start=0, stop=3)
  perf_df.columns = append_level(perf_df.columns, '')
  perf_df.columns = append_level(perf_df.columns, 't2')
  perf_df.columns.set_names(['hyper', 'metric', '', 't2'], inplace=True)
  auto_plot(perf_df, path_suffix=path_suffix + '_perf',
            axes_names=['hyper', 'metric', ''],
            other_names=['t2'])
  
  # plot statistics
  df = visualizer.stat_df.copy()
  
  df.columns = expand_level(df.columns)
  # # name0 1 2 3 -- obs0 conv2d act iqr
  # df = select(df, {'dataset_type': 'cifar10'})
  df.columns = merge_level(df.columns, start=5, stop=7)
  df = reindex(df, 5)
  df.columns = merge_level(df.columns, start=3, stop=5)
  
  df.columns = merge_level(df.columns, start=0, stop=3)
  df.columns.set_names(['hyper', 'layer', 'stat', 'winsize'], inplace=True)
  df = select(df,
              {'hyper': '(?:resnet10/cifar10/1.*3|vgg10/cifar10/1.*2)'})  # (?:resnet10/cifar10/.*3|vgg10/cifar10/.*2)
  
  paths = auto_plot(df, path_suffix=path_suffix + '_stat',
                    axes_names=('layer', 'stat', 'winsize'),
                    other_names=['hyper'])
  utils.merge_pdf(paths)
if __name__ == '__main__':
  from vis_utils import Visualizer
  
  visualizer = Visualizer(join='outer', stat_only=True, paranet_folder='stdtime')
  df = visualizer.perf_df
  arr, indexf = df2arr(df)
  print df.shape, arr.shape
  df2 = arr2df(arr, indexf)
  
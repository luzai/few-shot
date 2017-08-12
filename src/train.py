def run(model_type='vgg6', lr=1e-2, limit_val=True,
        dataset='cifar10', queue=None, runtime=1,
        with_bn=True, ):
  import utils
  import warnings
  warnings.filterwarnings("ignore")
  
  utils.init_dev(utils.get_dev())
  # utils.allow_growth()
  import tensorflow as tf, keras
  from callbacks import TensorBoard2
  from keras.callbacks import TensorBoard
  from datasets import Dataset
  from models import VGG, ResNet
  from configs import Config
  from loader import Loader
  from logs import logger
  import utils, configs
  # try:


  config = Config(epochs=utils.get_config('epochs'),
                  batch_size=256, verbose=2,
                  model_type=model_type,
                  dataset_type=dataset,
                  debug=False,
                  others={'lr'     : lr,
                          'runtime': runtime,
                          'with_bn': with_bn, },
                  clean_after=False)
  
  dataset = Dataset(config.dataset_type, debug=config.debug, limit_val=limit_val)
  if 'vgg' in model_type:
    model = VGG(dataset.input_shape, dataset.classes, config, with_bn=with_bn, with_dp=True)
  else:
    model = ResNet(dataset.input_shape, dataset.classes, config, with_bn=with_bn, with_dp=True)
  
  model.model.summary()
  model.model.compile(
      # keras.optimizers.rmsprop(lr=0.0001, decay=1e-6),
      keras.optimizers.sgd(lr, momentum=0.9),
      loss='categorical_crossentropy',
      metrics=['accuracy'])
  
  if queue is not None: queue.put([True])
  # todo callback monisotr fisrt epoch
  model.model.fit(dataset.x_train, dataset.y_train, batch_size=config.batch_size, epochs=config.epochs,
                  verbose=config.verbose,
                  validation_data=(dataset.x_test, dataset.y_test),
                  callbacks=[
                    TensorBoard2(tot_epochs=config.epochs,
                                 log_dir=config.model_tfevents_path,
                                 batch_size=config.batch_size,
                                 write_graph=True,
                                 write_grads=False,
                                 dataset=dataset,
                                 max_win_size=31 if utils.get_config()['dbg'] else 3,
                                 stat_only=True,
                                 batch_based=True
                                 ),
                    # TensorBoard(log_dir=config.model_tfevents_path)
                  ])
  if config.clean_after:
    Loader(path=config.model_tfevents_path).load()
  model.save()
  # except Exception as inst:
  #     print inst
  #     exit(100)


import multiprocessing as mp, time
from logs import logger
import logs, os.path as osp
import numpy as np
import utils, os, np_utils


utils.rm(utils.root_path + '/tfevents  ' + utils.root_path + '/output')
tasks = []
if utils.get_config()['dbg']:
  queue = mp.Queue()
  logger.error('!!! your are in dbg')
  run('vgg10', dataset='mnist', lr=1e-2)
  # run('resnet10', dataset='cifar10', lr=1e-3)
else:
  queue = mp.Queue()
  
  # grids = {'dataset': ['cifar10', ],
  #          'model_type': ['vgg10', ],
  #          'lr': np.logspace(-2, -3, 2),
  #          'queue': [queue, ],
  #          'runtime': [1, ]
  #          }
  
  grids = {'dataset'   : ['cifar10', ],
           'model_type': ['vgg6'],  # 'vgg16', , 'vgg19'
           'lr'        : np.logspace(-2, -3, 2),
           'queue'     : [queue, ],
           'runtime'   : [1, ],
           'with_bn'   : [True, False]
           }
  
  for grid in np_utils.grid_iter(grids):
    print grid
    p = mp.Process(target=run, kwargs=grid)
    p.start()
    tasks.append(p)
    _res = queue.get()
    logger.info('last task return {}'.format(_res))
    time.sleep(15)
  for p in tasks:
    p.join()

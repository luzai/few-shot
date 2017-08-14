def run(model_type='vgg6', limit_val=True,
        dataset='cifar10', queue=None, runtime=1,
        with_bn=True, optimizer={'name': 'sgd'}):
  import utils
  import warnings
  warnings.filterwarnings("ignore")
  
  utils.init_dev(utils.get_dev(ok=utils.get_config('gpu')))
  # utils.allow_growth()
  import tensorflow as tf, keras
  from callbacks import TensorBoard2, schedule
  from keras.callbacks import TensorBoard, LearningRateScheduler
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
                  others={'runtime'  : runtime,
                          'with_bn'  : with_bn,
                          'optimizer': optimizer,
                          },
                  )
  # if len(glob.glob(config.model_tfevents_path + '/*tfevents*')) >= 1:
  #   logger.info('exist ' + config.model_tfevents_path)
  #   if queue is not None: queue.put([True])
  #   exit(200)
  # else:
  #   logger.info('do not exist ' + config.model_tfevents_path)
  
  dataset = Dataset(config.dataset_type, debug=config.debug, limit_val=limit_val)
  if 'vgg' in model_type:
    model = VGG(dataset.input_shape, dataset.classes, config, with_bn=with_bn, with_dp=True)
  else:
    model = ResNet(dataset.input_shape, dataset.classes, config, with_bn=with_bn, with_dp=True)
  if optimizer['name'] == 'sgd':
    opt = keras.optimizers.sgd(optimizer['lr'], momentum=0.9)
  elif optimizer['name'] == 'rmsprop':
    opt = keras.optimizers.rmsprop(optimizer['lr'])
  else:
    opt = keras.optimizers.adam(optimizer['lr'])
  model.model.summary()
  model.model.compile(opt,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
  
  if queue is not None: queue.put([True])
  
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
                                 max_win_size=11,
                                 stat_only=True,
                                 batch_based=True
                                 ),
                    LearningRateScheduler(lambda epoch: schedule(epoch,
                                                                 x=optimizer.get('decay_epoch', None),
                                                                 y=optimizer.get('decay', None),
                                                                 init=optimizer['lr'],
                                                                 ),
                                          ),
                    # TensorBoard(log_dir=config.model_tfevents_path)
                  ])
  
  Loader(path=config.model_tfevents_path,stat_only=True).load(stat_only=True)
  model.save()
  # except Exception as inst:
  #     print inst
  #     exit(100)


import multiprocessing as mp, time
from logs import logger
import logs, os.path as osp
import numpy as np
import utils, os, legacy

utils.rm(utils.root_path + '/tfevents  ' + utils.root_path + '/output')
tasks = []

queue = mp.Queue()
grids = utils.get_config('grids')
for grid in utils.grid_iter(grids):
  print grid
  if utils.get_config('dbg'):
    logger.error('!!! your are in dbg')
    run(**grid)
    break
  else:
    p = mp.Process(target=run, kwargs=utils.dict_concat([grid, {'queue': queue}]))
    p.start()
    tasks.append(p)
    _res = queue.get()
    logger.info('last task return {}'.format(_res))
    time.sleep(15)
for p in tasks:
  p.join()

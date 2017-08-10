def run(model_type='vgg5', lr=1e-2,
        dataset='cifar10', queue=None,
        limit_val=True, optimizer='sgd', decay_epoch=None, decay=None
        ):
    import utils
    import warnings
    warnings.filterwarnings("ignore")

    utils.init_dev(utils.get_dev())
    utils.allow_growth()
    import tensorflow as tf, keras
    from keras.callbacks import TensorBoard, LearningRateScheduler
    from callbacks import schedule
    from datasets import Dataset
    from models import VGG, ResNet
    from configs import Config
    from loader import Loader
    from logs import logger

    # try:
    config = Config(epochs=201, batch_size=256, verbose=2,
                    model_type=model_type,
                    dataset_type=dataset,
                    debug=False,
                    others={'lr': lr, 'decay_epoch': decay_epoch, 'decay': decay},
                    clean_after=False)

    dataset = Dataset(config.dataset_type, debug=config.debug, limit_val=limit_val)
    if 'vgg' in model_type:
        model = VGG(dataset.input_shape, dataset.classes, config, with_bn=True, with_dp=True)
    else:
        model = ResNet(dataset.input_shape, dataset.classes, config, with_bn=True, with_dp=True)

    model.model.summary()
    model.model.compile(
        keras.optimizers.sgd(lr, momentum=0.9) if optimizer == 'sgd' else keras.optimizers.rmsprop(),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    if queue is not None: queue.put([True])
    model.model.fit(dataset.x_train, dataset.y_train, batch_size=config.batch_size, epochs=config.epochs,
                    verbose=config.verbose,
                    validation_data=(dataset.x_test, dataset.y_test),
                    callbacks=[
                        TensorBoard(log_dir=config.model_tfevents_path),
                        LearningRateScheduler(lambda epoch: schedule(epoch, x=decay_epoch, y=decay))
                    ])
    model.save()
    # except Exception as inst:
    #     print inst
    #     exit(100)


import multiprocessing as mp, time
from logs import logger
import logs
import numpy as np
import utils, os

utils.rm(utils.root_path + '/tfevents  ' + utils.root_path + '/output')
if os.path.exists('dbg'):
    logger.error('!!! your are in dbg')
    run('vgg10', dataset='cifar10', lr=1e-2)
else:
    queue = mp.Queue()
    tasks = []
    dataset = 'cifar10'
    for model_type in ['vgg10', 'resnet10', ]:  # 'vgg8', 'resnet8','vgg6', 'resnet6',
        for lr in np.logspace(-2, -3, 4):
            for epoch in [5, 35, 75, 105]:
                for decay in [2, 5, 10, 20]:
                    print dataset, model_type, lr
                    p = mp.Process(target=run, kwargs=dict(model_type=model_type, lr=lr,
                                                           dataset=dataset, queue=queue,
                                                           decay=decay,decay_epoch=epoch))
                    p.start()
                    tasks.append(p)
                    _res = queue.get()
                    logger.info('last task return {}'.format(_res))
                    time.sleep(15)

    for p in tasks:
        p.join()
def run(model_type='vgg5', lr=1e-2, limit_val=True, dataset='cifar10', queue=None):
    import utils
    import warnings
    warnings.filterwarnings("ignore")

    utils.init_dev(utils.get_dev())
    utils.allow_growth()
    import tensorflow as tf, keras
    from saver import TensorBoard2
    from keras.callbacks import TensorBoard
    from datasets import Dataset
    from models import VGG, ResNet
    from configs import Config
    from loader import Loader
    from logs import logger

    # try:
    config = Config(epochs=101, batch_size=256, verbose=2,
                    model_type=model_type,
                    dataset_type=dataset,
                    debug=False, others={'lr': lr}, clean_after=False)

    dataset = Dataset(config.dataset_type, debug=config.debug, limit_val=limit_val)
    if 'vgg' in model_type:
        model = VGG(dataset.input_shape, dataset.classes, config, with_bn=False, with_dp=True)
    else:
        model = ResNet(dataset.input_shape, dataset.classes, config, with_bn=False, with_dp=True)

    model.model.summary()
    # todo lr scheme
    # todo verify!! validation dataset size sensitivity
    model.model.compile(
        # keras.optimizers.rmsprop(lr=0.0001, decay=1e-6),
        # in fact rmsprop is much better at least on small dataset
        keras.optimizers.sgd(lr, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    if queue is not None: queue.put([True])
    model.model.fit(dataset.x_train, dataset.y_train, batch_size=config.batch_size, epochs=config.epochs,
                    verbose=config.verbose,
                    validation_data=(dataset.x_test, dataset.y_test),
                    callbacks=[
                        TensorBoard2(log_dir=config.model_tfevents_path,
                                     histogram_freq=5,
                                     batch_size=config.batch_size,
                                     write_graph=True,
                                     write_grads=False,
                                     dataset=dataset
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
import numpy as np
import utils

utils.rm('../tfevents  ../output')

# run('vgg6', dataset='cifar10', lr=1e-2)
# run('resnet10', dataset='cifar10', lr=1e-2)
# run('vgg6',1e-5)

queue = mp.Queue()
tasks = []
for dataset in ['cifar10', 'cifar100']:  # , 'cifar100'
    for model_type in [ 'vgg10', 'resnet10', ]:  # 'vgg8', 'resnet8','vgg6', 'resnet6',
        for lr in np.concatenate((np.logspace(-2, -3, 2), np.logspace(-1.5, -2.5, 0))):  # 10,1e-1, 1e-3, 1e-5
            print dataset, model_type, lr
            p = mp.Process(target=run, args=(model_type, lr, True, dataset, queue))
            p.start()
            tasks.append(p)
            _res = queue.get()
            logger.info('last task return {}'.format(_res))
            time.sleep(15)

for p in tasks:
    p.join()

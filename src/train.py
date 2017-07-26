def run(model_type='vgg5', lr=1e-2, limit_val=True):
    import utils
    import warnings
    warnings.filterwarnings("ignore")
    # todo report/observe speed
    # todo callback on iter end rather epoch
    utils.init_dev(utils.get_dev())
    utils.allow_growth()
    import tensorflow as tf, keras
    from saver import TensorBoard
    from datasets import Dataset
    from models import VGG
    from opts import Config
    from loader import Loader
    from log import logger

    config = Config(epochs=41, batch_size=256, verbose=2,
                    model_type=model_type,
                    dataset_type='cifar10',
                    debug=False, others={'lr': lr}, clean_after=True)

    dataset = Dataset(config.dataset_type, debug=config.debug, limit_val=limit_val)
    vgg = VGG(dataset.input_shape, dataset.classes, config.model_type, with_bn=False, with_dp=True, name=config.name)
    vgg.model.summary()
    # todo lr scheme
    # todo plataea auto detect and variant sample rate
    # todo verify!! validation dataset size sensitivity
    vgg.model.compile(
        # keras.optimizers.rmsprop(lr=0.0001, decay=1e-6),
        keras.optimizers.sgd(1e-2),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    import tensorflow as tf

    vgg.model.fit(dataset.x_train, dataset.y_train, batch_size=config.batch_size, epochs=config.epochs,
                  verbose=config.verbose,
                  validation_data=(dataset.x_test, dataset.y_test),
                  callbacks=[TensorBoard(log_dir=config.model_tfevents_path,
                                         histogram_freq=5,
                                         batch_size=config.batch_size,
                                         write_graph=True,
                                         write_grads=False,
                                         dataset=dataset
                                         )]
                  )
    if config.clean_after:
        Loader(path=config.model_tfevents_path).load()


import multiprocessing as mp, time
import subprocess

subprocess.call('rm -r ../tfevents ../output'.split())
subprocess.call('rm -r tfevents output'.split())
# run('vgg5')
tasks = []
for model_type in ['vgg5', 'vgg11', 'vgg19']:
    for lr in [1, 1e-2, 1e-5]:
        p = mp.Process(target=run, args=(model_type, lr))
        p.start()
        tasks.append(p)
        time.sleep(25)
for p in tasks:
    p.join()

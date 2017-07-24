def run(model_type='vgg5', n=None):
    import utils
    if n is None:
        n= utils.get_dev()
    utils.init_dev(n)
    utils.allow_growth()
    import tensorflow as tf,keras
    from saver import TensorBoard
    from datasets import Dataset
    from models import VGG
    from opts import Config
    from log import logger

    config = Config(epochs=301, batch_size=256, verbose=2,
                    model_type=model_type,
                    dataset_type='cifar10',
                    debug=False)

    dataset = Dataset(config.dataset_type, debug=config.debug)
    vgg = VGG(dataset.input_shape, dataset.classes, config.model_type, with_bn=False, with_dp=False)

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
                  # callbacks=[TensorBoard(log_dir=config.model_tfevents_path,
                  #                        histogram_freq=0,
                  #                        batch_size=config.batch_size,
                  #                        write_graph=True,
                  #                        write_grads=False,
                  #                        )]
                  )


import multiprocessing as mp

gpu = 0
run('vgg5')
# for model_type in ['vgg5', 'vgg11']:
#     p = mp.Process(target=run, args=(model_type, gpu))
#     gpu += 1
#     p.start()
#     p.join()
# run('vgg5')

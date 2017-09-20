from easydict import EasyDict as edict
from utils import root_path
import utils

cifar10 = edict(data_dir=root_path + '/data/cifar10',
                batch_size=128,
                log_dir=root_path + '/output/cifar10',
                checkpoint_path='../models/resnet101/resnet_v2_101.ckpt',
                init_lr=0.1,
                lr_decay_per_steps=45000,
                lr_decay=0.1
                )

cifar10_eval = edict(
    data_dir=root_path + '/data/cifar10',
    batch_size=cifar10.batch_size,
    log_dir=cifar10.log_dir + '_eval',
    checkpoint_dir=cifar10.log_dir,
    eval_interval_secs=60,
    num_evals=10000 // cifar10.batch_size + 1,
)

cifar100 = edict(data_dir='../data/cifar100',
                 batch_size=128,
                 log_dir='../output/multiloss-fixl-loss20-3',
                 checkpoint_path='../models/resnet50/resnet_v2_50.ckpt',
                 multi_loss=False,
                 beta=1.,
                 gamma=1.,
                 init_lr=0.1,
                 lr_decay_per_steps=45000,
                 lr_decay=0.1,
                 interval=800,
                 )

cifar100_eval = edict(data_dir='../data/cifar100',
                      batch_size=cifar100.batch_size,
                      log_dir=cifar100.log_dir + '_eval',
                      checkpoint_dir=cifar100.log_dir,
                      num_evals=10000 // cifar100.batch_size + 1,
                      eval_interval_secs=60,
                      multi_loss=False,
                      beta=1.,
                      gamma=1.
                      )

imagenet = edict(
    init_lr=0.01,
    lr_decay_per_steps=40000,  # 100000
    lr_decay=0.1,
    interval=1800, )
imagenet.data_dir = utils.root_path + '/data/imagenet600'
imagenet.nclasses = 669  # 7460 # 669
imagenet.log_dir = '../output/imgnet600-4'
imagenet.nimgs = 81589 * 9  # 5543684  # 81589*9

imagenet.batch_size = 32
imagenet.num_clones = 8
imagenet.checkpoint_path = utils.root_path + '/models/resnet101/resnet_v2_101.ckpt'

imagenet.nsteps = None
# imagenet.log_dir = '../output/' + utils.randomword(10),
# todo method?
# imagenet.beta = 1.
# imagenet.gamma = 1.

imgnet_eval = edict(
    data_dir=imagenet.data_dir,
    batch_size=imagenet.batch_size,
    log_dir=imagenet.log_dir + '_eval',
    checkpoint_dir=imagenet.log_dir,
    eval_interval_secs=60,
    num_evals=int(imagenet.nimgs / 9. / 32) + 1,
    nclasses=imagenet.nclasses,
    nimgs=imagenet.nimgs // 9
)

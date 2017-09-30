"""ResNet-v2(Identity mapping)"""
import os
import sys

parrots_home = os.environ.get('PARROTS_HOME') or '/home/wangxinglu/.local/lib/python2.7/site-packages/parrots'
if not parrots_home:
    raise EnvironmentError(
        'The environment variable "PARROTS_HOME" is not set.')
sys.path.append(os.path.join(parrots_home))

if os.environ.get('PARROTS_HOME') is None:
    from parrots.dnn.modules import ModuleProto, GModule
    from parrots.dnn.layerprotos import (Convolution, FullyConnected, Pooling,
                                         Sum, Softmax, BN, ReLU, Dropout,
                                         SoftmaxWithLoss, Accuracy)
else:
    from parrots.dnn.modules import ModuleProto, GModule
    from parrots.dnn.layerprotos import (Convolution, FullyConnected, Pooling,
                                         Sum, Softmax, BN, ReLU, Dropout,
                                         SoftmaxWithLoss, Accuracy)


class Bottleneck(ModuleProto):
    def __init__(self,
                 out_channels,
                 stride=1,
                 preact_branch='single',
                 shortcut_type='identity'):
        assert preact_branch in ['single', 'both', 'none']
        assert shortcut_type in ['identity', 'conv']
        self.out_channels = out_channels
        self.stride = stride
        self.preact_branch = preact_branch
        self.shortcut_type = shortcut_type

    def construct(self, m):
        stride = self.stride
        out_channels = self.out_channels
        x = m.input_slot('x')
        if self.preact_branch == 'single':
            r = x.to(BN(), name='bn1')
            r = r.to(ReLU(), inplace=True, name='relu1')
        else:
            if self.preact_branch == 'both':
                x = x.to(BN(), name='bn1')
                x = x.to(ReLU(), inplace=True, name='relu1')
            r = x

        r = r.to(Convolution(1, out_channels, bias=False), name='conv1')
        r = r.to(BN(), name='bn2')
        r = r.to(ReLU(), inplace=True, name='relu2')
        r = r.to(Convolution(
            3, out_channels, stride=stride, pad=1, bias=False),
            name='conv2')
        r = r.to(BN(), name='bn3')
        r = r.to(ReLU(), inplace=True, name='relu3')
        r = r.to(Convolution(1, out_channels * 4, bias=False), name='conv3')
        if self.shortcut_type == 'conv':
            x = x.to(Convolution(
                1, out_channels * 4, stride=stride, bias=False),
                name='shortcut')
        x = m.vars(x, r).to(Sum(), name='sum')

        m.output_slots = x.name


class BasicBlock(ModuleProto):
    def __init__(self,
                 out_channels,
                 stride=1,
                 preact_branch='single',
                 shortcut_type='identity'):
        assert preact_branch in ['single', 'both', 'none']
        assert shortcut_type in ['identity', 'conv']
        self.out_channels = out_channels
        self.stride = stride
        self.preact_branch = preact_branch
        self.shortcut_type = shortcut_type

    def construct(self, m):
        stride = self.stride
        out_channels = self.out_channels
        x = m.input_slot('x')
        if self.preact_branch == 'single':
            r = x.to(BN(), name='bn1')
            r = r.to(ReLU(), inplace=True, name='relu1')
        else:
            if self.preact_branch == 'both':
                x = x.to(BN(), name='bn1')
                x = x.to(ReLU(), inplace=True, name='relu1')
            r = x

        r = r.to(Convolution(
            3, out_channels, stride=stride, pad=1, bias=False),
            name='conv1')
        r = r.to(BN(), name='bn2')
        r = r.to(ReLU(), inplace=True, name='relu2')
        r = r.to(Convolution(3, out_channels, pad=1, bias=False), name='conv2')

        if self.shortcut_type == 'conv':
            x = x.to(Convolution(
                1, out_channels, stride=stride, bias=False),
                name='shortcut')
        x = m.vars(x, r).to(Sum(), name='sum')

        m.output_slots = x.name


def create_model(depth=101, input_size=224, num_classes=1000, name=None):
    cfg = {
        18: (BasicBlock, [(2, 64), (2, 128), (2, 256), (2, 512)]),
        34: (BasicBlock, [(3, 64), (4, 128), (6, 256), (3, 512)]),
        50: (Bottleneck, [(3, 64), (4, 128), (6, 256), (3, 512)]),
        101: (Bottleneck, [(3, 64), (4, 128), (23, 256), (3, 512)]),
        '101_10k': (Bottleneck, [(3, 64), (4, 256), (23, 512), (3, 2560)]),
        152: (Bottleneck, [(3, 64), (8, 128), (36, 256), (3, 512)]),
        200: (Bottleneck, [(3, 64), (24, 128), (36, 256), (3, 512)]),
    }

    assert depth in cfg

    if name is None:
        name = 'resnet-{}'.format(depth)
    main = GModule(name)
    inputs = {
        'data': 'float32({}, {}, 3, _)'.format(input_size, input_size),
        'label': 'uint32(1, _)'
    }
    main.input_slots = tuple(inputs.keys())

    x = main.var('data')
    x = x.to(Convolution(7, 64, stride=2, pad=3, bias=False), name='conv1')
    x = x.to(BN(), name='bn1')
    x = x.to(ReLU(), inplace=True, name='relu1')
    x = x.to(Pooling('max', 3, stride=2), name='pool1')

    block, params = cfg[depth]

    for i, (num, out_channels) in enumerate(params):
        stride = 1 if i == 0 else 2
        preact_branch = 'none' if i == 0 else 'both'
        x = x.to(block(out_channels, stride, preact_branch, 'conv'),
                 name='res{}a'.format(i + 2))
        for j in range(1, num):
            x = x.to(block(out_channels, 1), name='res{}b{}'.format(i + 2, j))

    num_stages = len(params) + 1
    x = x.to(BN(), name='bn{}'.format(num_stages))
    x = x.to(ReLU(), inplace=True, name='relu{}'.format(num_stages))
    x = x.to(Pooling('ave', 7), name='pool{}'.format(num_stages))
    x = x.to(Dropout(0.5), inplace=True, name='dropout')
    x = x.to(FullyConnected(
        num_classes,
        # w_policy={'init': 'gauss(0.01)',
        #           'decay_mult': '1'},
        # b_policy={'init': 'fill(0)',
        #           'decay_mult': '1',
        #           'lr_mult': '1'}
    ),
        name='cls')
    x.to(Softmax(), name='prob')
    main.vars(x, 'label').to(SoftmaxWithLoss(), name='loss')
    main.vars(x, 'label').to(Accuracy(1), name='accuracy_top1')
    main.vars(x, 'label').to(Accuracy(5), name='accuracy_top5')
    model = main.compile(inputs=inputs, seal=False)
    model.add_flow('main',
                   inputs.keys(), ['loss', 'accuracy_top1', 'accuracy_top5'],
                   ['loss'])
    model.seal()
    return main.compile(inputs=inputs)


def creat_flatten_model(depth=101, input_size=224, num_classes=1000, name=None):
    cfg = {
        18: (BasicBlock, [(2, 64), (2, 128), (2, 256), (2, 512)]),
        34: (BasicBlock, [(3, 64), (4, 128), (6, 256), (3, 512)]),
        50: (Bottleneck, [(3, 64), (4, 128), (6, 256), (3, 512)]),
        101: (Bottleneck, [(3, 64), (4, 128), (23, 256), (3, 512)]),
        152: (Bottleneck, [(3, 64), (8, 128), (36, 256), (3, 512)]),
        200: (Bottleneck, [(3, 64), (24, 128), (36, 256), (3, 512)]),
    }

    assert depth in cfg

    if name is None:
        name = 'resnet-{}'.format(depth)
    main = GModule(name)
    inputs = {
        'data': 'float32({}, {}, 3, _)'.format(input_size, input_size),
        'label': 'uint32(1, _)'
    }
    main.input_slots = tuple(inputs.keys())

    x = main.var('data')
    x = x.to(Convolution(7, 64, stride=2, pad=3, bias=False), name='conv1')
    x = x.to(BN(), name='bn1')
    x = x.to(ReLU(), inplace=True, name='relu1')
    x = x.to(Pooling('max', 3, stride=2), name='pool1')

    block, params = cfg[depth]

    for i, (num, out_channels) in enumerate(params):
        stride = 1 if i == 0 else 2
        preact_branch = 'none' if i == 0 else 'both'
        x = x.to(block(out_channels, stride, preact_branch, 'conv'),
                 name='res{}a'.format(i + 2))
        for j in range(1, num):
            x = x.to(block(out_channels, 1), name='res{}b{}'.format(i + 2, j))

    num_stages = len(params) + 1
    x = x.to(BN(), name='bn{}'.format(num_stages))
    x = x.to(ReLU(), inplace=True, name='relu{}'.format(num_stages))

    x = x.to(Convolution(3, 1200, pad=0, bias=False, stride=2), name='luzai.conv')
    x = x.to(BN(), name='luzai.bn')
    x = x.to(ReLU(), inplace=True, name='luzai.relu')

    # x = x.to(Pooling('ave', 7), name='pool{}'.format(num_stages))

    x = x.to(Dropout(0.5), inplace=True, name='dropout')
    x = x.to(FullyConnected(
        num_classes,
        # w_policy={'init': 'gauss(0.01)',
        #           'decay_mult': '1'},
        # b_policy={'init': 'fill(0)',
        #           'decay_mult': '1',
        #           'lr_mult': '1'}
    ),
        name='luzai.cls')
    x.to(Softmax(), name='prob')
    main.vars(x, 'label').to(SoftmaxWithLoss(), name='loss')
    main.vars(x, 'label').to(Accuracy(1, ), name='accuracy_top1')
    main.vars(x, 'label').to(Accuracy(5, ), name='accuracy_top5')
    model = main.compile(inputs=inputs, seal=False)
    model.add_flow('main',
                   inputs.keys(), ['loss', 'accuracy_top1', 'accuracy_top5'],
                   ['loss'])
    model.seal()
    return main.compile(inputs=inputs)


if __name__ == '__main__':
    model = create_model(101)
    print(model.to_yaml_text())
    with open('res1k.yaml', 'w') as f:
        print >> f, model.to_yaml_text()

    model = create_model(depth='101_10k', num_classes=10000)
    print(model.to_yaml_text())
    with open('res10k.yaml', 'w') as f:
        print >> f, model.to_yaml_text()

    model = creat_flatten_model(depth=101, num_classes=10000)
    print(model.to_yaml_text())
    with open('res10k-flatten.yaml', 'w') as f:
        print >> f, model.to_yaml_text()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable-all

import os, sys

parrots_home = os.environ.get('PARROTS_HOME',None)
if not parrots_home:
    raise EnvironmentError("The environment variable 'PARROTS_HOME' must be set.")
sys.path.append(os.path.join(parrots_home, 'parrots', 'python'))
from parrots import dnn
from parrots.dnn.modules import ModuleProto, GModule
from parrots.dnn import layerprotos
from parrots.dnn.layerprotos import Convolution, FullyConnected, GlobalPooling, Pooling, Concat, Sum
from parrots.dnn.layerprotos import BN, ReLU, Dropout, SoftmaxWithLoss, Accuracy
print os.environ['LD_LIBRARY_PATH']

# exit(-1)

class BasicBlock(ModuleProto):
    def __init__(self, out, shrink=False):
        self.out = out
        self.shrink = shrink

    def construct(self, m):
        x = m.input_slot('x')
        stride = 2 if self.shrink else 1
        r = (x.to(Convolution(3, self.out, stride=stride, pad=1, bias=False), name='conv1')
             .to(BN(), name='bn1')
             .to(ReLU(), inplace=True, name='relu1')
             .to(Convolution(3, self.out, pad=1, bias=False), name='conv2')
             .to(BN(), name='bn2'))
        if self.shrink:
            x = (x.to(Convolution(1, self.out, stride=2), name='shrink')
                 .to(BN(), name='shrinkbn'))

        x = (m.vars(x, r)
             .to(Sum(), name='sum')
             .to(ReLU(), inplace=True, name='relu'))

        m.output_slots = x.name


def create_model(name="ResNet56", input=32, dropout=0.5, depth=20, nclass=62, dataset='cifar'):
    """
    - dataset: "cifar", depth 20, 32, 44, 56, 110, 1202
    """
    main = GModule(name)
    main.input_slots = ('data', 'label')
    inputs = {
        'data': 'float32({}, {}, 3, _)'.format(input, input),
        'label': 'uint32(1, _)'
    }

    if dataset == 'cifar':
        assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110, 1202'
        n = (depth - 2) / 6

        x = (main.var('data')
             .to(Convolution(3, 16, pad=1, bias=False), name='conv1'))

        for i in range(n):
            x = x.to(BasicBlock(16, False), name='a{}'.format(i))
        for i in range(n):
            x = x.to(BasicBlock(32, i == 0), name='b{}'.format(i))
        for i in range(n):
            x = x.to(BasicBlock(64, i == 0), name='c{}'.format(i))

        x = (x.to(BN(), name='last')
             .to(ReLU(), inplace=True, name='relu'))
        x = x.to(GlobalPooling('ave'), name='pool')
        x = (x.to(Dropout(dropout), name='dropout')
             .to(FullyConnected(nclass), name='fc'))

    else:
        raise ValueError('unsupported dataset: ' + dataset)

    main.vars('fc', 'label').to(SoftmaxWithLoss(), name='loss')
    main.vars('fc', 'label').to(Accuracy(1), name='accuracy_top1')
    if nclass > 100:
        main.vars('fc', 'label').to(Accuracy(5), name='accuracy_top5')

    return main.compile(inputs=inputs)


if __name__ == '__main__':
    model = create_model(input=32, dataset='cifar', depth=56, nclass=100)
    print(model.to_yaml_text())

    with open('resnet.yaml', 'w') as f:
        print >> f, model.to_yaml_text()

    os.system(os.path.join(parrots_home, 'parrots', 'tools', 'visdnn') + ' resnet.yaml -o resnet.dot > /dev/null')

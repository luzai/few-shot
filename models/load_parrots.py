#!/usr/bin/env python
# coding: utf-8

import sys
import h5py
import yaml
import re
import argparse

parser = argparse.ArgumentParser(description="Convert Parrots model to Caffe style")
parser.add_argument("parrots_spec", default=None)
parser.add_argument("caffe_spec", default=None)
parser.add_argument("--parrots_weights", default=None)
parser.add_argument("--caffe_weights", default=None)
parser.add_argument("--caffe_root", default='.',
                    help="path to the Caffe installation, Python interfaces must be enabled")

args = parser.parse_args()

sys.path.append(args.caffe_root)
import caffe
from caffe import layers as L, params as P

# This dictionary maps Parrots specific layer names to their Caffe counterparts
name_mapping = {
    'FullyConnected': 'InnerProduct'
}

expr_parser = re.compile('([^\s]+)\s?=\s?([^()]+)\((.*)\)')
shape_parser = re.compile('\s?([^()]+)\((.*)\)')


def parse_parrots_expr(expr):
    m = re.match(expr_parser, expr)
    out_str, op, in_str = m.groups()
    out_list = [x.strip() for x in out_str.strip().split(',')]
    in_list = [x.strip() for x in in_str.strip().split(',')]

    in_blob_list = [x for x in in_list if not x.startswith('@')]
    param_list = [x[1:] for x in in_list if x.startswith('@')]

    return out_list, op, in_blob_list, param_list


def parse_shape(expr):
    m = re.match(shape_parser, expr)
    dtype, shape_str = m.groups()
    shape = tuple([int(x.strip()) if x.strip() != '_' else 1 for x in shape_str.split(',')][::-1])
    return shape


parrots_spec = yaml.load(open(args.parrots_spec))
if args.parrots_weights is not None:
    parrots_weights = h5py.File(args.parrots_weights)
else:
    parrots_weights = None

parrots_layers = parrots_spec['layers']


## Since Parrots use flat config structures, we have to manually construct their Caffe conterparts
## These are param builder for Caffe specs, you can tweak them to have other effect or add new ones for your own layers
def bn_param_func(name, attrs, params, weights_dict):
    """
    Set up the caffe-style BN layer params
    """
    configs = {
        'bn_param': dict(slope_filler=dict(type='constant', value=1.0),
                         bias_filler=dict(type='constant', value=0.0)),
        'param': [dict(lr_mult=1.0, decay_mult=0.0), dict(lr_mult=1.0, decay_mult=0.0)]
    }

    if weights_dict is not None:
        scale = weights_dict[params[0] + '@value'][...].copy()
        bias = weights_dict[params[1] + '@value'][...].copy()
        stats = weights_dict[params[2] + '@value'][...]
        mean = stats[:stats.size / 2].copy()
        var = stats[stats.size / 2:].copy()
        weights = (scale, bias, mean, var)
    else:
        weights = None
    return configs, weights


def conv_param_func(name, attrs, params, weights_dict):
    configs = {
        'param': [dict(lr_mult=1.0, decay_mult=1.0 * (i + 1)) for i, p in enumerate(params)]
    }

    if len(params) > 2:
        raise ValueError("Convolution layer can have at most 2 params: " + name)

    conv_params = dict(weight_filler=dict(type='xavier'), bias_filler=dict(value=0.0, type='constant'))
    conv_params.update(attrs)

    if len(params) == 1:
        conv_params['bias_term'] = False

    configs['convolution_param'] = conv_params

    weights = [weights_dict[x + '@value'] for x in params] if weights_dict is not None else None

    return configs, weights


def fc_param_func(name, attrs, params, weights_dict):
    configs = {
        'param': [dict(lr_mult=1.0, decay_mult=1.0 * (i + 1)) for i, p in enumerate(params)]
    }

    if len(params) > 2:
        raise ValueError("FullyConnected layer can have at most 2 params: " + name)

    fc_params = dict(weight_filler=dict(type='xavier'), bias_filler=dict(value=0.0, type='constant'))
    fc_params.update(attrs)

    if len(params) == 1:
        conv_params['bias_term'] = False

    configs['inner_product_param'] = fc_params

    weights = [weights_dict[x + '@value'] for x in params] if weights_dict is not None else None

    return configs, weights


def pooling_param_func(name, attrs, params, weights_dict):
    pooling_params = {}
    for k, v in attrs.items():
        if k != 'mode':
            pooling_params[k] = v
        else:
            if v == 'max':
                pooling_params['pool'] = P.Pooling.MAX
            elif v == 'ave':
                pooling_params['pool'] = P.Pooling.AVE
    configs = {
        'pooling_param': pooling_params
    }
    return configs, None


def eltwise_param_func(name, attrs, params, weights_dict):
    eltwise_params = {}
    coeff = []
    for k, v in attrs.items():
        if k == 'op':
            if v == 'sum':
                eltwise_params['operation'] = P.Eltwise.SUM
            elif v == 'prod':
                eltwise_params['operation'] = P.Eltwise.PROD
            elif v == 'max':
                eltwise_params['operation'] = P.Eltwise.MAX
        elif type(k) == int:
            coeff.append(v)
        else:
            eltwise_params[k] = v

    if len(coeff) > 0:
        eltwise_params['coeff'] = coeff
    configs = {
        'eltwise_param': eltwise_params
    }
    return configs, None


def plain_param_func_gen(param_name):
    """
    This function is a function wrapper that generates default param builders.
    It throws all `attrs` in parrots to the `param_name` in Caffe's proto
    """

    def param_func(name, attrs, params, weights_dict):
        if attrs is None:
            return dict(), None
        configs = {
            param_name: attrs
        }
        return configs, None

    return param_func


no_param_func = lambda x, y, z, f: (dict(), None)

# Register param builders to the table
pdict = {}
pdict['Dropout'] = plain_param_func_gen('dropout_param')
pdict['SoftmaxWithLoss'] = plain_param_func_gen('softmax_param')
pdict['Accuracy'] = plain_param_func_gen('accuracy_param')

pdict['ReLU'] = no_param_func
pdict['Concat'] = no_param_func

pdict['BN'] = bn_param_func
pdict['Eltwise'] = eltwise_param_func
pdict['FullyConnected'] = fc_param_func
pdict['Convolution'] = conv_param_func
pdict['Pooling'] = pooling_param_func


def build_caffe_layer(top_blob_dict, save_weights_dict, layer_info, param_func_dict,
                      weights_dict=None, bottom_layer=False,
                      name_mapping=dict()):
    attrs = layer_info.setdefault('attrs')
    layer_out, layer_op, layer_in, layer_params = parse_parrots_expr(layer_info['expr'])
    layer_name = layer_info['id']

    # check input rename
    for i in xrange(len(layer_in)):
        if layer_in[i] in var_rename_dict:
            layer_in[i] = var_rename_dict[layer_in[i]]

    # check output
    if len(layer_out) > 1:
        raise ValueError("Cannot support more than 1 output for a layer: " + layer_name)

    # build params
    layer_configs, layer_weights = param_func_dict[layer_op](layer_name, attrs, layer_params, weights_dict)
    layer_configs['name'] = layer_name

    # support data input
    if bottom_layer:
        layer_configs['bottom'] = layer_in
        in_tops = []
    else:
        trailing = False
        in_tops = []
        abs_in = []
        for x in layer_in:
            try:
                in_blob = top_blob_dict[x]
            except KeyError:
                in_blob = None
            if not trailing:
                if in_blob is not None:
                    in_tops.append(in_blob)
                else:
                    abs_in.append(x)
                    trailing = True
            else:
                if in_blobs is not None:
                    raise ValueError("Non-trailing data input at layer: " + layer_name)
                else:
                    abs_in.append(x)
        if trailing:
            layer_configs['bottom'] = abs_in

    # identify in-place layer
    in_place = False
    if len(layer_in) == 1 and len(layer_out) == 1 and layer_in[0] == layer_out[0]:
        layer_configs['in_place'] = True
        in_place = True
    elif layer_in[0] == layer_out[0] and len(layer_in) != len(layer_out):
        # we have to dismantle the in-place computation
        tmp = layer_out[0]
        var_rename_dict[tmp] = tmp + '_dup'
        layer_out[0] = tmp + '_dup'

    op_name = name_mapping.setdefault(layer_op, layer_op)

    caffe_top = getattr(L, op_name)(*in_tops, **layer_configs)

    top_blob_dict[layer_out[0]] = caffe_top

    save_weights_dict[layer_name] = layer_weights

    return layer_out[0] if not in_place else layer_name, caffe_top


caffe_net_spec = caffe.NetSpec()

top_dict = {}
save_weights_dict = {}
var_rename_dict = {}
for i in xrange(len(parrots_layers)):
    entry, top = build_caffe_layer(top_dict, save_weights_dict, parrots_layers[i], pdict,
                                   parrots_weights, i == 0, name_mapping)
    setattr(caffe_net_spec, entry, top)

netp = caffe_net_spec.to_proto()

parrot_input_list = parrots_spec['inputs']

for item in parrot_input_list:
    netp.input.append(item['id'])
    s = caffe.io.caffe_pb2.BlobShape()
    s.dim.extend(parse_shape(item['spec']))
    netp.input_shape.extend([s])

# output the Caffe prototxt
open(args.caffe_spec, 'w').write(str(netp))

# try construct the network in Caffe, as a final check
caffe_net = caffe.Net(args.caffe_spec, caffe.TEST)

# output the caffemodel if requested
if args.caffe_weights is not None:
    for k, v in save_weights_dict.items():
        if v is None:
            continue
        for i in xrange(len(v)):
            caffe_net.params[k][i].data[...] = v[i]
    caffe_net.save(args.caffe_weights)

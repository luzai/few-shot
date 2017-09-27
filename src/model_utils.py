from utils import *


def plt_per_cls_acc(lb, pred):
    from sklearn.metrics import confusion_matrix
    conf = confusion_matrix(lb, pred, np.arange(0, 1000)).astype(float)
    cnts, bounds, _ = plt.hist(lb, bins=1000, range=(-0.5, 1000 - 0.5))
    cnts[cnts == 0] = 1
    markerline, stemlines, baseline = plt.stem(np.diag(conf))
    _ = plt.setp(baseline, 'color', 'b')
    _ = plt.setp(stemlines, 'color', 'b')
    _ = plt.step(markerline, 'color', 'b')
    return np.diag(conf) / cnts.astype(float)
    # plt.plot(cnts*np.diag(conf)/cnts)


def load_model():
    import pyparrots
    import string, os
    from pyparrots.env import Environ
    import pyparrots.dnn as dnn
    import yaml

    os.chdir('/home/wangxinglu/prj/few-shot/models/res101.img1k.longtail3/')
    session_file = './session.yaml'
    model_file = "./model.yaml"
    param_file = "snapshots/iter.best.parrots"

    mapping = dict(gpu='2:4', bs=8 * 2, )

    # read model file
    with open(model_file) as fin:
        model_text = fin.read()
    # read session file
    with open(session_file, 'r') as fcfg:
        cfg_templ_in = fcfg.read()
    cfg_templ = string.Template(cfg_templ_in)

    cfg_text = cfg_templ.substitute(mapping)

    cfg_text = yaml.dump(yaml.load(cfg_text))

    # yaml.load(model_text)
    # yaml.load(cfg_text)

    # create model
    model = dnn.Model.from_yaml_text(model_text)
    # create session
    session = dnn.Session.from_yaml_text(model, cfg_text)

    session.setup()

    f = session.flow('val')
    f.load_param(param_file)
    return f


def extract_feature(f):
    lbs, datas, fcs = [], [], []
    ind = 0
    while ind < 602017:
        f.feed()
        f.forward()
        lbs.append(f.data('label').value())
        #     datas.append(f.data('data').value())
        fcs.append(f.data('fc').value())
        ind += lbs[-1].shape[-1]

    lb = np.concatenate(lbs, axis=-1)
    fc = np.concatenate(fcs, axis=-1)

    pred = np.argmax(fc, axis=0)

    lb = lb.reshape((lb.shape[-1],))

    return lb, pred


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


def strip_dp(l):
    res = [l_ for l_ in l if 'dropout' not in l_['expr'].lower()]
    # print l_['expr'].lower()
    return res


def get_param_mapping(beforef, afterf):
    spec1 = yaml.load(open(beforef, 'r'))
    spec2 = yaml.load(open(afterf, 'r'))
    mapping = {}
    layers1, layers2 = strip_dp(spec1['layers']), strip_dp(spec2['layers'])
    assert len(layers1) == len(layers2), str(len(layers1)) + ' ' + str(len(layers2))
    for l1, l2 in zip(layers1, layers2):
        #         print l1,l2
        #         break
        params_b = parse_parrots_expr(l1['expr'])[-1]
        params_a = parse_parrots_expr(l2['expr'])[-1]
        assert len(params_b) == len(params_a), l1['expr'] + l2['expr']
        for a, b in zip(params_a, params_b):
            mapping[a] = b
    return mapping


model_spec_file = root_path + '/models/meta/model.yaml'
# root_path + '/models/res101.img1k/session.yaml'
model_file = '/mnt/gv7/16winter/16winter/ijcai/resnet101/model.parrots'
spec_f2 = root_path + '/models/meta/res1k.yaml'
mapping = get_param_mapping(spec_f2, model_spec_file)

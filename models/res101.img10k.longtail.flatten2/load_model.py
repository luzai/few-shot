import sys, os

sys.path.append(os.path.join('', 'parrots/python'))


def load_model(path):
    import parrots
    import string, os
    from parrots.env import Environ
    import parrots.dnn as dnn
    import yaml

    os.chdir(path)
    session_file = './session.yaml'

    # read session file
    mapping = dict(HOME='/home/wangxinglu')

    with open(session_file, 'r') as fcfg:
        cfg_templ_in = fcfg.read()
    cfg_templ = string.Template(cfg_templ_in)

    cfg_text = cfg_templ.substitute(mapping)
    cfg_text = yaml.dump(yaml.load(cfg_text))
    cfg = yaml.load(cfg_text)
    cfg['flows'][1]['val']['feeder']['pipeline'][0]['attr']['shuffle'] = False
    cfg['flows'][1]['val']['batch_size'] = 108
    cfg['flows'][1]['val']['devices'] = 'gpu(4:6)'
    cfg['flows'][0]['train']['batch_size'] = 108
    cfg['flows'][0]['train']['devices'] = 'gpu(4:6)'
    cfg_text = yaml.dump(cfg)

    model_file = cfg["model"]["yaml"]
    param_file = "snapshots/iter.best.parrots"

    # read model file
    with open(model_file) as fin:
        model_text = fin.read()

    # create model
    model = dnn.Model.from_yaml_text(model_text)
    # create session
    session = dnn.Session.from_yaml_text(model, cfg_text)
    with open('session.bak.yaml', 'w') as fh:
        fh.write(cfg_text)

    session.setup()

    f = session.flow('val')
    f.load_param(param_file)
    return f


def save_fea(f, name):
    import h5py
    ff = h5py.File(name)
    ind = 0
    while ind < 568165:
        f.feed()
        f.forward()
        lbs = f.data('label').value()
        ff['lbs/' + str(ind)] = lbs
        ff['fcs/' + str(ind)] = f.data('prob').value()
        ind += lbs[-1].shape[-1]

    ff.close()


def load_fea(name, return_fc=False):
    import h5py, numpy as np
    f = h5py.File(name, 'r')

    orders = f['fcs'].keys()

    lbs = [f['lbs/' + l].value for l in orders]
    fcs = [f['fcs/' + l].value for l in orders]

    lb = np.concatenate(lbs, axis=-1)
    fc = np.concatenate(fcs, axis=-1)

    pred = np.argmax(fc, axis=0)
    pred = pred.reshape((pred.shape[-1],))
    lb = lb.reshape((lb.shape[-1],))

    f.close()
    if return_fc:
        return pred, lb, fc
    else:
        return pred, lb


# f = load_model('/home/wangxinglu/prj/few-shot/models/res101.img10k.longtail.flatten2/')
# print f
#
# save_fea(f, 'flatten2.h5')
# f.release()

pred, lb = load_fea('flatten2.h5')

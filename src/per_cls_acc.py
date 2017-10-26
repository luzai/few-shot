from utils import *
from metadata import *
from model_utils import *
from gen_imglst import *
from keras import backend as K
from keras.engine.topology import Layer


@chdir_to_root
def load_model(path):
    import parrots
    import string, os
    from parrots.env import Environ
    import parrots.dnn as dnn
    import yaml

    os.chdir(path)
    session_file = './session.yaml'

    # read session file
    mapping ={'HOME':'/home/wangxinglu'}
    with open(session_file, 'r') as fcfg:
        cfg_templ_in = fcfg.read()
    cfg_templ = string.Template(cfg_templ_in)
    cfg_text = cfg_templ.substitute(mapping)
    print cfg_text
    # cfg_text = cfg_templ_in

    cfg_text = yaml.dump(yaml.load(cfg_text))
    cfg_text = yaml.load(cfg_text)
    cfg_text['flows'][1]['val']['feeder']['pipeline'][0]['attr']['shuffle'] = False
    device = 'host'  # 'host'  #
    cfg_text['flows'][0]['train']['devices'] = device
    cfg_text['flows'][1]['val']['devices'] = device
    cfg_text['flows'][1]['val']['batch_size'] = 20
    cfg_text['flows'][0]['train']['batch_size'] = 20
    model_file = cfg_text["model"]["yaml"]
    cfg_text =  yaml.dump(cfg_text)
    param_file = "snapshots/iter.best.parrots"

    # read model file
    with open(model_file) as fin:
        model_text = fin.read()

    # create model
    model = dnn.Model.from_yaml_text(model_text)
    # print(model.to_yaml_text())
    # create session
    session = dnn.Session.from_yaml_text(model, cfg_text)

    session.setup()
    print 'success!!'
    f = session.flow('val')
    f.load_param(param_file)
    return f

@timeit()
def get_prediction(path=None, name=None):
    if name is None:
        name = path.split('/')[-1] + '.h5'

    if not osp.exists(path + '/snapshots/iter.best.parrots'):
        utils.ln(path=path + '/snapshots/iter.latest.parrots', to_path=path + '/snapshots/iter.best.parrots')
    f = load_model(path)
    utils.rm(name)
    ff = h5py.File(name)
    ind = 0
    while ind < 568106:
        print ind
        if ind % 500 == 0:
            print ind
        if ind > 1000:
            break
        f.feed()
        f.forward()
        lbs = f.data('label').value()
        ff['lbs/' + str(ind)] = lbs
        ff['fcs/' + str(ind)] = f.data('prob').value()
        ind += lbs[-1].shape[-1]

    ff.close()
    f.release()


path = '/home/wangxinglu/prj/few-shot/models/'

import multiprocessing as mp

for d in [
    # 'res101.img10k.two_level',
    'res101.img10k.longtail.flatten.2048.retrain',
]:
    p = path + d
    n = d + '.h5'
    print p, n
    get_prediction(p, n)
    print 'ok'

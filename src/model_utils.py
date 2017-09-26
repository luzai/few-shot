from utils import *


def per_cls_acc(lb, pred):
    from sklearn.metrics import confusion_matrix
    conf = confusion_matrix(lb, pred, np.arange(0, 1000)).astype(float)
    cnts, bounds, _ = plt.hist(lb, bins=1000, range=(-0.5, 1000 - 0.5))
    cnts[cnts==0]=1
    markerline, stemlines, baseline = plt.stem(np.diag(conf))
    _ = plt.setp(baseline, 'color', 'b')
    _ = plt.setp(stemlines, 'color', 'b')
    _ = plt.step(markerline,'color','b')
    return np.diag(conf) / cnts.astype(float)
    # plt.plot(cnts*np.diag(conf)/cnts)


def load_model():
    import pyparrots
    import string, os
    from pyparrots.env import Environ
    import pyparrots.dnn as dnn
    import yaml

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

    # uncommend following line to load parameter
    f = session.flow('val')

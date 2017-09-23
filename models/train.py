import pyparrots
import string,os
from pyparrots.env import Environ
import pyparrots.dnn as dnn

session_file= './session.yaml'
model_file = "./model.yaml"
param_file = "/mnt/gv7/16winter/16winter/ijcai/resnet101/model.parrots"

mapping=dict(gpu='2:4',
            bs=8*2,
            )

# read model file
with open(model_file) as fin:
    model_text = fin.read()

# read session file
with open(session_file, 'r') as fcfg:
    cfg_templ = fcfg.read()
cfg_templ = string.Template(cfg_templ)

cfg_text = cfg_templ.substitute(mapping)
# create model
model = dnn.Model.from_yaml_text(model_text)
# create session
session = dnn.Session.from_yaml_text(model, cfg_text)

session.setup()

# uncommend following line to load parameter
session.flow('val').load_param(param_file)

with session.flow('val') as f:
    f.forward()

# f=session.flow('val')
f.forward()
list(f.input_ids())
list(f.output_ids())


f.data('loss').value()
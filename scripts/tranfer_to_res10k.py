%cd src
from utils import *
import utils
%load_ext autoreload
# %reload_ext autoreload
%autoreload 2
%matplotlib inline
import matplotlib
matplotlib.style.use('ggplot')

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# import parrots
from model_utils import *

specf2 = root_path + '/models/meta/res1k.yaml'
modelf2 = root_path + '/models/resnet101/model.1k.parrots'
specf3=root_path+'/models/meta/res10k.yaml'
modelf3=root_path+'/models/resnet101/model.10k.parrots'

def get_params(specf, modelf):
    f = h5py.File(modelf)
    params_ = {k: v for k, v in f.iteritems()}

    spec = yaml.load(open(specf, 'r'))
    params = collections.OrderedDict()
    for l in spec['layers']:
        for ll in parse_parrots_expr(l['expr'])[-1]:
            if ll + '@value' in params_:
                params[ll] = params_[ll + '@value'][...].copy()
    f.close()
    return params



t=get_params(specf2,modelf2)
t.keys()[:5]

tt=get_params(specf3,modelf3)
tt.keys()[:5]

def yield_weight(t,tt):
    for name,name2,w1,w2 in zip(t.keys(),tt.keys(),t.values(),tt.values()):
        w1=w1.squeeze()
        w2=w2.squeeze()
        if w1.shape != w2.shape:
            assert name==name2
            yield name,w1,w2

def normalize(x):
    return x/np.sum(x)
t=np.array([0.76,	0.12,	0.12]) * np.array([1.35,0.7,1.35])
normalize(t)


t=np.array([0.76,	0.12,	0.12]) * np.array([1.5,0,1.5])
normalize(t)


ttt=tt.copy()
for name,w1,w2 in yield_weight(t,tt):
    mul=[]
    for m1,m2 in zip(w1.shape,w2.shape):
        mul.append(m2//m1)
    ttt[name]=np.tile(w1,mul) / float(np.prod(np.sqrt(mul)))
ttt.keys()[:5]
!rm $modelf3
f=h5py.File(modelf3)

for k,v in ttt.iteritems():
    f[k]=v

f.close()

f=h5py.File(modelf3)

len(f.keys())
f.keys()[:5]
f.close()

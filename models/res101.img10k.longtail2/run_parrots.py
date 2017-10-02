import subprocess, os, numpy as np

env = os.environ.copy()

subprocess.call('rm -rf .lock')

p = subprocess.Popen('parrots -d train session.yaml',
                     shell=True,
                     stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE,
                     env=env
                     )

out, err = p.communicate()
while np.in1d(['core', 'dump'], [out, err]):
    p = subprocess.Popen('parrots -d resume .')
    out, err = p.communicate()

import subprocess, os, numpy as np

env = os.environ.copy()

subprocess.call('rm -rf .lock'.split())
subprocess.call('rm -rf log.txt'.split())

# p = subprocess.Popen('parrots -d train session.yaml',
p = subprocess.Popen('parrots -d resume . ',
                     shell=True,
                     stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE,
                     env=env
                     )

out, err = p.communicate()
print 'out', out, 'err', err


def isin(l1, l2):
    # if not isinstance(l2, list):
    #     l2 = [l2]
    flag = False
    for l_ in l1:
        if l_ in l2: flag = True
    return flag


while isin(['backtrace', 'core dump', 'exception'], (out + err).lower()):
    p = subprocess.Popen('parrots -d resume .')
    out, err = p.communicate()
    print 'out', out, 'err', err

import signal
import os
import contextlib
import subprocess
import logging
import warnings
import subprocess, os, numpy as np

env = os.environ.copy()

subprocess.call('rm -rf .lock'.split())
subprocess.call('rm -rf log.txt'.split())


@contextlib.contextmanager
def process_fixture(shell_args):
    proc = subprocess.Popen(shell_args,
                            preexec_fn=os.setsid,
                            shell=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            env=env
                            )
    try:
        yield proc
    except Exception as inst:
        print '-->fail', inst
    finally:
        proc.terminate()
        proc.wait()

        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except OSError as e:
            warnings.warn(e)


def isin(l1, l2):
    # if not isinstance(l2, list):
    #     l2 = [l2]
    flag = False
    for l_ in l1:
        if l_ in l2: flag = True
    return flag


if __name__ == '__main__':
    while True:
        with process_fixture('parrots -d resume . ') as proc:
            print('pid %d' % proc.pid)
            out, err = proc.communicate()
            print 'out', out, 'err', err
            if not isin(['backtrace', 'core dump', 'exception'], (out + err).lower()):
                break

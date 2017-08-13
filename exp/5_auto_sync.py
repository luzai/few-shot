import subprocess, time,glob
import os

home = os.path.expanduser('~')
my_env = os.environ.copy()
print home


def scp(src, dest):
  cmd =('scp -r ' + src + ' ' + dest)
  print cmd
  task = subprocess.Popen(cmd.split(),
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          env=my_env)
  print task.communicate()
  # subprocess.call(('scp -r ' + src + ' ' + dest).split())


# todo

while True:
  print 'hi'
  for path in glob.glob('/home/wangxinglu/prj/Perf_Pred/src/*.py'):
    scp(path, 'yxli:~/prj/Perf_Pred/src')
  print 'success'
  time.sleep(3)

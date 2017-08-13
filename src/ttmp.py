im_l = []
import glob
import numpy as np
import matplotlib.pylab as plt
from scipy.misc import *

for img in glob.glob('../data/2003/*/*/*/*')[:500]:
  # print 'ji'
  im = imread(img, mode='RGB')
  im = imresize(im, (80, 80))
  # print im.shape
  im_l.append(im)

for img in glob.glob('/home/yxli/ext/train2014/*')[:3500]:
  # print 'aa'
  im = imread(img, mode='RGB')
  im = imresize(im, (80, 80))
  # print im.shape
  im_l.append(im)

im_f = np.array(im_l)
x_test = im_f
x_test=x_test[np.random.permutation(len(x_test))]

gl = 0
for i_ in range(20):
  t = x_test[gl:gl + 160]
  fig, axes = plt.subplots(16, 8, figsize=(8, 16))
  axes = axes.flatten()
  for ind, axis in enumerate(axes):
    axis.imshow(t[ind])
    axis.axis('off')
  print 'ok'
  gl += 161
  fig.savefig(str(i_) + '.png')
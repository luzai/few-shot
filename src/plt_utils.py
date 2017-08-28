import plotly
import tensorflow as tf
# plotly.tools.set_credentials_file(username='luzai', api_key='4VdMbq5oQVSLgKR7GpUB')
from keras.utils import to_categorical
import plotly.graph_objs as go
import plotly.plotly as py

import numpy as np


def scatter(x, y, c):
  
  x = x.flatten()
  y = y.flatten()
  c = np.array(c)
  c = c.flatten()
 
  trace1 = go.Scatter(
      x=x,
      y=y,
      name='scatter',
      mode='markers',
      marker=dict(
          size='5',
          color=c,  # set color equal to a variable
          colorscale='Viridis',
          showscale=True
      )
  )
  data = [trace1]
  return data
  

def plot(data,static=True):
  layout = go.Layout(title='', width=600, height=600)
  if not static:
    res = py.iplot(dict(data=data,layout=layout), filename='scatter-plot-with-colorscale')
  else:
    fig = go.Figure(data=data, layout=layout)
    py.image.ishow(fig)
    res = None
  return res

def line(x,y,c=None,name=''):
  data=[go.Scatter(x=x,y=y,
                   name=name,
                   mode='maker+line',
                   marker=dict(
                       size='10',
                       color=c
                       # colorscale='Viridis',
                       # showscale=False
                   ),
                   line=dict(
                       color=c,
                       # colorscale='Viridis',
                       # showlegend=False
                   ))
        ]
  return data

def gen_disk(smpls=(10, 10)):
  r = np.linspace(0.1, 2, smpls[0])
  theta = np.linspace(0+0.1, np.pi * 2+0.1, smpls[1])
  R, Theta = np.meshgrid(r, theta)
  X, Y = np.empty_like(R), np.empty_like(R)
  for i in range(R.shape[0]):
    for j in range(R.shape[1]):
      X[i, j] = R[i, j] * np.cos(Theta[i, j])
      Y[i, j] = R[i, j] * np.sin(Theta[i, j])
  return X, Y


def hinge_loss(pred_in, true_in):
  true_in = to_categorical(true_in)
  # print true_in
  pred = tf.placeholder(tf.float32)
  true = tf.placeholder(tf.float32)
  loss = tf.losses.hinge_loss(true, pred)
  with tf.Session() as sess:
    loss_out = sess.run(loss, feed_dict={pred: pred_in, true: true_in})
  
  return loss_out


def mse_loss(pred, true):
  # true = to_categorical(true)
  loss = np.square(pred - true)
  if len(loss.shape)==2:
    loss = loss.sum(axis=1).mean()
  else:
    loss =loss.sum()
  return loss

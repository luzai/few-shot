from loader import *
from datasets import *
from vis_utils import *
from utils import *
from stats import *
from model_utils import *
from plt_utils import *
from logs import logger
import logs, datasets, vis_utils, loader

utils.init_dev(utils.get_dev())
utils.allow_growth()

def matrix_symmetric(x):
    return (x + tf.transpose(x, [0, 2, 1])) / 2

def get_eigen_K(x, square=False):
    """
    Get K = 1 / (sigma_i - sigma_j) for i != j, 0 otherwise

    Parameters
    ----------
    x : tf.Tensor with shape as [..., dim,]

    Returns
    -------

    """
    if square:
        x = tf.square(x)
    res = tf.expand_dims(x, 1) - tf.expand_dims(x, 2)
    res += tf.eye(tf.shape(res)[1])
    res = 1 / res
    res -= tf.eye(tf.shape(res)[1])
    
    # Keep the results clean
    res = tf.where(tf.is_nan(res), tf.zeros_like(res), res)
    res = tf.where(tf.is_inf(res), tf.zeros_like(res), res)
    return res

@tf.RegisterGradient('Svd')
def gradient_svd(op, grad_s, grad_u, grad_v):
    """
    Define the gradient for SVD
    References
        Ionescu, C., et al, Matrix Backpropagation for Deep Networks with Structured Layers

    Parameters
    ----------
    op
    grad_s
    grad_u
    grad_v

    Returns
    -------
    """
    s, u, v = op.outputs
    v_t = tf.transpose(v, [0, 2, 1])
    
    with tf.name_scope('K'):
        K = get_eigen_K(s, True)
    inner = matrix_symmetric(K * tf.matmul(v_t, grad_v))
    
    # Create the shape accordingly.
    u_shape = u.get_shape()[1].value
    v_shape = v.get_shape()[1].value
    
    # Recover the complete S matrices and its gradient
    eye_mat = tf.eye(v_shape, u_shape)
    realS = tf.matmul(tf.reshape(tf.matrix_diag(s), [-1, v_shape]), eye_mat)
    realS = tf.transpose(tf.reshape(realS, [-1, v_shape, u_shape]), [0, 2, 1])
    
    real_grad_S = tf.matmul(tf.reshape(tf.matrix_diag(grad_s), [-1, v_shape]), eye_mat)
    real_grad_S = tf.transpose(tf.reshape(real_grad_S, [-1, v_shape, u_shape]), [0, 2, 1])
    
    dxdz = tf.matmul(u, tf.matmul(2 * tf.matmul(realS, inner) + real_grad_S, v_t))
    return dxdz

def Svd(x):
    s, u, v = tf.svd(x, full_matrices=True)
    return s, u, v

def get_s(fake):
    fake = fake.squeeze()
    u, s, v = np.linalg.svd(fake, full_matrices=True)
    return to_int(s)

fake = gen_fake()
print get_s(fake)

with tf.Session() as sess:
    x = tf.placeholder(np.float32, shape=fake.shape)
    shape_x = x.get_shape().as_list()
    x_reshaped = tf.reshape(x, (np.prod(shape_x[:-1]), shape_x[-1]))
    x_reshaped = tf.expand_dims(x_reshaped, axis=0)
    s, u, v = Svd(x_reshaped)
    summ=tf.summary.histogram('single_value',s)
    s_min = tf.reduce_min(s)
    s = s - s_min
    loss = tf.reduce_sum(tf.square(s)) + tf.reduce_sum(0. * u) + tf.reduce_sum(0. * v)
    
    # print tf.gradients([s], [x])[0].eval()
    grad = tf.gradients([loss], [x])[0]
    # print tf.gradients([v], [x])[0].eval()
    
    tf.global_variables_initializer().run()
    graph = tf.get_default_graph()
    writer=tf.summary.FileWriter('./tf', graph=graph)

    for i in range(150):
        grad_np,summ_str = sess.run([grad,summ], {x: fake})
        fake -= grad_np * 1e-2
        print get_s(fake)
        writer.add_summary(summ_str,i)
writer.flush()
writer.close()

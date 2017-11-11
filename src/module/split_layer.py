import numpy as np, copy
from parrots.data import DataSpec
from parrots.dnn import PythonLayer, register_pylayer
from parrots import dnn
import logging, colorlog, os

logging.root.setLevel(logging.DEBUG)


def set_stream_logger(log_level=logging.INFO):
    sh = colorlog.StreamHandler()
    sh.setLevel(log_level)
    sh.setFormatter(
        colorlog.ColoredFormatter(
            '%(asctime)s %(filename)s [line:%(lineno)d] %(log_color)s%(levelname)s%(reset)s %(message)s'))
    logging.root.addHandler(sh)


def set_file_logger(work_dir=None, log_level=logging.DEBUG):
    work_dir = work_dir or os.getcwd()
    fh = logging.FileHandler(os.path.join(work_dir, 'log.txt'))
    fh.setLevel(log_level)
    fh.setFormatter(
        logging.Formatter('%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s'))
    logging.root.addHandler(fh)


set_stream_logger()
set_file_logger()


class MySplit(PythonLayer):
    '''This layer can take 1 to 65535 variables as inputs,
       concat them at ``concat_dim`` dimension to one variable
       as output. By default, it will concat at channel, i.e. at dim 2
    '''

    def _init_extra(self, split_dim=2):
        self.split_dim = split_dim

    @property
    def num_inputs(self):
        return 1

    @property
    def num_outputs(self):
        return 2

    def infer_spec(self, ins):
        '''
        ``ins`` is the list of input DataSpec
        '''

        # check shape of ins are valid
        assert self.split_dim < ins[0].ndims

        out1 = list(ins[0].shape)
        out2 = copy.deepcopy(out1)
        # logging.info('input is %s ', out1)

        chls = out1[self.split_dim]
        chls1 = chls // 2
        chls2 = chls - chls1
        self.split_at = chls1
        out1[self.split_dim] = chls1
        out2[self.split_dim] = chls2
        logging.info('out is %s %s', out1, out2)

        return [DataSpec.array(ins[0].elemtype, out1), DataSpec.array(ins[0].elemtype, out2)]

    def forward_cpu(self, bottom_values):
        '''
        ``bottom_values``   is the list of input numpy array followed
                            by learnable parameters.
        note:   this layer has no parameters, so the ``bottom_values``
                are just inputs
        '''

        '''return list of numpy array, which is the result
         of top values.
        This layer has only one output, so it return a list
        only contains one array.
        '''
        # from IPython import embed
        # embed()
        # logging.info(".. %s", bottom_values[0].shape)
        split_at = bottom_values[0].shape[self.split_dim] // 2
        # logging.info(".. %s", bottom_values[0][..., : split_at].shape)
        top1, top2 = bottom_values[0][:, :, :split_at, :], bottom_values[0][:, :, split_at:, :]
        logging.info('%s %s', top1.shape, top2.shape)
        top1, top2 = np.asfortranarray(top1), np.asfortranarray(top2)
        return [top1, top2]

    def backward_cpu(self, bottom_values, top_values, top_grads):
        '''
        ``bottom_values``   is same as what in forward_cpu;
        ``top_values``      is the list of top value;
        ``top_grads``       is the list of top gradient.
        '''

        '''return list of numpy array, which is the result of
        bottom gradients.
        '''
        logging.info('%s ', top_grads[0].flags['F_CONTIGUOUS'])
        return np.concatenate(top_grads, axis=self.split_dim)

        # try:
        #     dnn.register_pylayer(MySplit)
        #     logging.info('register once')
        # except:
        #     logging.error('register again')

import theano.tensor as T

from theano_extensions import conv2d_same
from init import weight_and_bias_init

class CGRU(object):
    """docstring for CGRU"""
    def __init__(self, filter_size=(3, 3), width=4, embedding=24, \
        num_relaxation=6, name=''):
        self.filter_size = filter_size
        self.width = width
        self.embedding = embedding
        self.num_relaxation = num_relaxation
        self.filter_shape = (embedding, embedding) + filter_size
        self.name = name

        self.W_gate, self.b_gate = weight_and_bias_init(\
            shape=self.filter_shape, name='gate_%s' % name, n=num_relaxation)
        self.W_switch, self.b_switch = weight_and_bias_init(\
            shape=self.filter_shape, name='switch_%s' % name, n=num_relaxation)
        self.W_rec, self.b_rec = weight_and_bias_init(\
            shape=self.filter_shape, name='rec_%s' % name, n=num_relaxation)

    @property
    def params(self):
        return [self.W_gate, self.b_gate, self.W_switch, \
                self.b_switch, self.W_rec, self.b_rec]

    def __call__(self, s, k):
        idx = k % self.num_relaxation
        conv_gate = conv2d_same(s, self.W_gate[idx], self.filter_size)
        u = T.nnet.hard_sigmoid(conv_gate + \
            self.b_gate[idx].dimshuffle('x', 0, 'x', 'x'))

        conv_switch = conv2d_same(s, self.W_switch[idx], self.filter_size)
        r = T.nnet.hard_sigmoid(conv_switch + \
            self.b_switch[idx].dimshuffle('x', 0, 'x', 'x'))

        conv_rec = conv2d_same(r * s, self.W_rec[idx], self.filter_size)
        return u * s + (1. - u) * T.tanh(conv_rec + \
            self.b_rec[idx].dimshuffle('x', 0, 'x', 'x'))
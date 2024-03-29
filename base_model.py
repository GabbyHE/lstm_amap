from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.layers import base


def root_mean_squared_error(labels, preds):
    total_size = tf.to_float(tf.size(labels))
    return tf.sqrt(tf.reduce_sum(tf.square(labels - preds)) / total_size)


def mean_absolute_error(labels, preds):
    total_size = 1. * tf.to_float(tf.size(labels))
    return tf.reduce_sum(tf.abs(labels - preds)) / total_size


def mean_squared_error(labels, preds):
    total_size = tf.to_float(tf.size(labels))
    return tf.reduce_sum(tf.squared_difference(labels, preds)) / total_size


class BaseModel(base.Layer):
    def __init__(self, hps, mode='train'):
        self.phs = {}  # placeholders
        self.hps = hps
        self.mode = mode
        self.lambda_l2_reg = tf.constant(hps.lambda_l2_reg, dtype=tf.float32)

        with tf.variable_scope('inputs'):
            # the input of global spatial attention, [batch_size, n_steps_encoder, n_links]
            self.phs['encoder_inputs'] = tf.placeholder(tf.float32,
                                                       [None, hps.n_steps_encoder,
                                                           hps.n_links],
                                                       name='encoder_inputs')
        
        with tf.variable_scope('groundtruth'):
            # Ground truth, [batch_size, n_steps_decoder, n_output_decoder], if no multi-task, n_output_decoder = 1
            self.phs['labels'] = tf.placeholder(tf.float32,
                                                [None, hps.n_steps_decoder,
                                                    hps.n_output_decoder],
                                                name='labels')
        self.phs['preds'] = None


    def build(self):
        pass

    @property
    def get_metric(self):
        metric_list = [root_mean_squared_error(self.phs['labels'], self.phs['preds']),
                       mean_absolute_error(self.phs['labels'], self.phs['preds'])]
        return metric_list

    def get_loss(self):
        pass

    def get_l2reg_loss(self):
        pass

    @property
    def loss(self):
        with tf.variable_scope('Loss'):
            return self.get_loss() + self.get_l2reg_loss()

    @property
    def train_op(self):
        pass

    def init(self, sess):
        sess.run(tf.global_variables_initializer())

    def summary(self, hps):
        pass
        
    def mod_fn(self):
        pass

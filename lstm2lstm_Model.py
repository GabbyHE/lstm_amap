# -*- coding: utf-8 -*-
"""
# Created on July 6, 2018
@author zhixianghe
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from base_model import BaseModel
from tensorflow.contrib import rnn
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest
from six.moves import xrange
from six.moves import zip
from lstm2lstm_utils import Linear


def input_transform(encoder_inputs, decoder_inputs, n_links, n_steps_encoder, n_steps_decoder, n_output_decoder):

    # transform the inputs from global view into encoder_inputs
    # print(encoder_inputs.shape)
    _encoder_inputs = tf.transpose(encoder_inputs, [1, 0, 2])
    _encoder_inputs = tf.reshape(_encoder_inputs, [-1, n_links])
    _encoder_inputs = tf.split(_encoder_inputs, n_steps_encoder, 0) #segment a row into small batches
    # print('encoder_inputs[0][0].shape:\t', _encoder_inputs[0][0].get_shape())

    encoder_inputs=_encoder_inputs
    # since we have only one travel time (traffic speed) series,
    # attention_input (local features) is SAME as the series which we are going to predict

    # transform the variables into listgs as the input of different function
    _decoder_inputs = tf.transpose(decoder_inputs, [1, 0, 2])
    _decoder_inputs = tf.reshape(_decoder_inputs, [-1, n_output_decoder])
    _decoder_inputs = tf.split(_decoder_inputs, n_steps_decoder, 0)


    # not useful when the loop function is employed
    decoder_inputs=[tf.zeros_like(
            _decoder_inputs[0], dtype=tf.float32, name="GO")] + _decoder_inputs[:-1]



    return encoder_inputs, decoder_inputs

def grid_spatial_cnn_model(grid_features, cnn_output_size): # dynamical spatial features
        """ Model function for CNN. """
        # Input layer
        input_layer = grid_features
        # input_layer = tf.reshape(grid_features, [-1, 8, 8, 1])
        # print(input_layer)
        # Convolutional layer #1
        filter1 = tf.get_variable('spatial_weights', [3, 3, 1, 64], \
                    initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), dtype=tf.float32)
        conv1 = nn_ops.conv2d(input_layer, filter1, strides = [1,1,1,1], padding = 'SAME')
        ## We shall be using max-pooling.
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=conv1,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='SAME')
        # Dense layer, also called fully connected layer
        pool_flat =  tf.reshape(layer, [-1, layer.get_shape()[1] * layer.get_shape()[2] * layer.get_shape()[3]])
        with vs.variable_scope("GridOutputProjection"):
            dense = tf.contrib.layers.fully_connected(pool_flat, cnn_output_size, activation_fn=tf.nn.relu)
        return dense

def spatial_static_cnn_model(static_attr, cnn_output_size):
    """ Model function for CNN. """
    # Input layer
    input_layer = tf.reshape(static_attr, [1, 8, 8, 5])
    # Convolutional layer #1
    filter1 = tf.get_variable('static_attr_weights', [3, 3, 5, 32], \
                initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), dtype=tf.float32)
    conv1 = nn_ops.conv2d(input_layer, filter1, strides = [1,1,1,1], padding = 'SAME')
    ## We shall be using max-pooling.
    layer = tf.nn.max_pool(value=conv1,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME')
    # Dense layer, also called fully connected layer
    pool_flat = tf.reshape(layer, [1, -1])
    with vs.variable_scope("static_GridOutputProjection"):
        static_dense = tf.contrib.layers.fully_connected(pool_flat, cnn_output_size, activation_fn = tf.nn.relu)

    return static_dense

def static_attr_fcl_model(static_attr, fcl_output_size):
    """ Model function for FCL on static link attr. """

    pool_flat = tf.reshape(static_attr, [1, -1])
    with vs.variable_scope("static_OutputProjection"):

        output = tf.contrib.layers.fully_connected(pool_flat, 4*fcl_output_size, activation_fn=None)
        output = tf.contrib.layers.batch_norm(output)
        output = tf.nn.relu(output)

        output = tf.contrib.layers.fully_connected(output, 2*fcl_output_size, activation_fn=None)
        output = tf.contrib.layers.batch_norm(output)
        output = tf.nn.relu(output)

        output = tf.contrib.layers.fully_connected(output, fcl_output_size, activation_fn=None)
        output = tf.contrib.layers.batch_norm(output)
        output = tf.nn.relu(output)

    return output

class LSTM_Model(BaseModel):
    """docstring for LSTM_Model"""
    def __init__(self, hps, mode='train'):
        super(LSTM_Model, self).__init__(hps, mode)
        preds=self.mod_fn()
        self.phs['preds'] = preds
        self.phs['loss'] = self.get_loss()  # see at eq.[11]
        tf.add_to_collection('loss', self.phs['loss'])
        self.phs['train_op'] = self.train_op()
        self.phs['summary'] = self.summary()

    def general_encoder(self, encoder_inputs, cell, output_size=None, dtype=dtypes.float32, scope=None):
        # check inputs
        if not encoder_inputs:
            raise ValueError(
                "Must provide at least 1 input to attention encoder.")
        if output_size is None:
            output_size=cell.output_size

        batch_size=array_ops.shape(encoder_inputs[0])[0]

        # how to get initial state
        initial_state_size=array_ops.stack([batch_size, output_size])
        initial_state_one=[array_ops.zeros(
            initial_state_size, dtype=dtype) for _ in xrange(2)]
        initial_state=[
            initial_state_one for _ in range(len(cell._cells))]
        state=initial_state

        # cell = tf.nn.rnn_cell.GRUCell(2)
        outputs=[]
        i=0
        # print ('encoder_inputs:\t', encoder_inputs)
        for s_inp in zip(encoder_inputs):
            # print(s_inp)

            if i>0:
                vs.get_variable_scope().reuse_variables()
            cell_output, state=cell(s_inp[0], state)#

            #output projection
            with vs.variable_scope("OutputProjection"):
                output=cell_output
            outputs.append(output)
            i+=1
        # print('...............................state')
        # print(state)
        # print('...............................outputs')
        # print(outputs)

        return outputs, state

    def temporal_attention(self,
                           decoder_inputs,
                           grid_features,
                           decoder_external_inputs,
                           s_attn_weights,
                           initial_state,
                           attention_states,
                           cell,
                           static_attr_dense,
                           loop_function=None,
                           external_flag=0,
                           s_attn_flag = 0,
                           t_attn_flag=0,
                           grid_flag=0,
                           static_attr_flag=0,
                           output_size=None,
                           dtype=tf.float32,
                           scope=None,
                           initial_state_attention=False,
                           ):
        """
        Temporal attention in LSTM_Model

        """

        # print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        # if self.hps.s_attn_flag=='1':
        #     print('s_attn_flag')

        if not decoder_inputs:
            raise ValueError(
                "Please give at leat 1 input to temporal attention decoder.")
        if not decoder_external_inputs:
            raise ValueError(
                "Please give at least 1 ext_input to temporal attention decoder.")

        if output_size is None:
            output_size=cell.output_size
            print('============================cell_output_size:\t', output_size)

        # implement of temporal attention
        with vs.variable_scope(
            scope or "temporal_attn", dtype=dtype) as scope:
            dtype=scope.dtype

            batch_size=array_ops.shape(decoder_inputs[0])[0]
            t_attn_length=attention_states.get_shape()[1].value
            if t_attn_length is None:
                t_attn_length=array_ops.shape(attention_states)[1]
            t_attn_size=attention_states.get_shape()[2].value
            hidden=array_ops.reshape(attention_states,
                                    [-1, t_attn_length, 1, t_attn_size])
            # Size of state vector for attention
            t_attn_vec_size=t_attn_size
            w=vs.get_variable(
                "Attn_Wd", [1,2,t_attn_size, t_attn_vec_size]) # W_d
            hidden_feature=nn_ops.conv2d(
                hidden, w, [1, 1,1,1], "SAME") #W_d * h_o
            v=vs.get_variable("Attn_v",[t_attn_vec_size]) # v_d
            state = initial_state

            def temporal_attention_weight(query):

                """
                    Put attention masks on self_hidden using self_hidden_features and query.
                """

                if nest.is_sequence(query):
                    query_list=nest.flatten(query)

                    for q in query_list:

                        ndims=q.get_shape().ndims
                        if ndims:
                            assert ndims==2
                    query = array_ops.concat(query_list, 1)


                # print('*************8***********state:\t%s' %  tf.shape(state[0]))
                with vs.variable_scope("Attn_Wpd"):
                    # Linear map
                    y=Linear(query, t_attn_vec_size, True)
                    y=array_ops.reshape(
                        y, [-1,1,1,t_attn_vec_size])

                    # Attention mask is a softmax of v_d^{\top} * tanH(...)
                    s=math_ops.reduce_sum(v*math_ops.tanh(hidden_feature + y + 1e-10),
                                           [2, 3])

                    a=nn_ops.softmax(s)
                    d=math_ops.reduce_sum(
                        array_ops.reshape(a, [-1,t_attn_length, 1, 1])* hidden, [1,2])
                return array_ops.reshape(d, [-1, t_attn_size])

            if initial_state_attention:
                t_attn=temporal_attention_weight(inital_state) # temporal attention vector
            else:
                batch_attn_size=array_ops.stack([batch_size, t_attn_size])
                t_attn=array_ops.zeros(batch_attn_size, dtype=dtype)
                t_attn.set_shape([None, t_attn_size])

            i=0
            outputs=[]
            prev=None


            for inp, ext_inp in zip(decoder_inputs, decoder_external_inputs):
                # print(inp)
                if i>0:
                    vs.get_variable_scope().reuse_variables()
                # If loop_function is set, we use it instead of decoder_inputs.
                if loop_function is not None and prev is not None:
                    # with vs.get_variable_scope("loop_function", reuse=True):
                    with vs.variable_scope("loop_function", reuse=True):

                        inp=loop_function(prev, i)

                # print('----------inp')
                # print(inp)
                # Merge input and previous attentions into one vector of the right size.
                # print(inp.get_shape())
                input_size = inp.get_shape().with_rank(2)[1]
                if input_size.value is None:
                    raise ValueError(
                        "Could not infer input size from input: %s" % inp.name)
                grid_flag = 0
                # we map the concatenation to shape [batch_size, input_size]
                if external_flag and t_attn_flag:
                    x = Linear([inp] + [ext_inp] + [t_attn], input_size, True)
                elif t_attn_flag:
                    x = Linear([inp] + [t_attn], input_size, True)
                else:
                    x = Linear([inp], input_size, True)

                cell_output, state = cell(x, state) # query: cell state
                static_dense= tf.tile(static_attr_dense, [batch_size, 1]) #self attention vector

                # Run the attention mechanism.
                if i == 0 and initial_state_attention:
                    with vs.variable_scope(vs.get_variable_scope(), reuse=True):
                        t_attn = temporal_attention_weight(state)
                else:
                    t_attn = temporal_attention_weight(state)

                with vs.variable_scope("AttnOutputProjection"):
                    if t_attn_flag and static_attr_flag:
                        fclInput = tf.concat([cell_output, t_attn, static_dense], 1)

                    elif static_attr_flag:
                        fclInput = tf.concat([cell_output, t_attn], 1)

                    else:
                        fclInput = cell_output

                    output = tf.contrib.layers.fully_connected(fclInput, output_size, activation_fn=tf.nn.relu)
                    # output = tf.contrib.layers.batch_norm(output)
                    # output = tf.nn.relu(output)
                # print('---=============================arrived at loop_function')
                if loop_function is not None:
                    prev = output
                outputs.append(output)
                i += 1
        return outputs, state

    def general_decoder(self, decoder_inputs, initial_state, cell, loop_function=None, output_size=None, dtype=tf.float32, scope=None, initial_state_attention=False):
        """

        """

        if not decoder_inputs:
            raise ValueError(
                "Please give at leat 1 input to temporal attention decoder.")

        if output_size is None:
            output_size=cell.output_size
            print('============================cell_output_size:\t', output_size)

        # implement of temporal attention
        with vs.variable_scope(
            scope or "general_decoder", dtype=dtype) as scope:
            dtype=scope.dtype
            batch_size=array_ops.shape(decoder_inputs[0])[0]
            state = initial_state

            i=0
            outputs=[]
            prev=None
            # print ('decoder_inputs:\t', decoder_inputs)
            for inp in zip(decoder_inputs):
                # print(inp)
                if i>0:
                    vs.get_variable_scope().reuse_variables()
                # # If loop_function is set, we use it instead of decoder_inputs.
                # if loop_function is not None and prev is not None:
                #     # with vs.get_variable_scope("loop_function", reuse=True):
                #     with vs.variable_scope("loop_function", reuse=True):

                #         inp=loop_function(prev, i)
                # x = tf.reshape(inp[0], [-1, self.hps.n_links])
                # print('decoder cell inputs:\t', inp)
                cell_output, state = cell(inp[0], state) # query: cell state

                with vs.variable_scope("AttnOutputProjection"):

                    output = tf.contrib.layers.fully_connected(cell_output, output_size, activation_fn=tf.nn.relu)

                if loop_function is not None:
                    prev = output
                outputs.append(output)
                i += 1
        return outputs, state

    def _loop_function(self, prev, _):
        """loop function used in the decoder to generate the next inupt"""
        return tf.matmul(prev, self.phs['w_out']) + self.phs['b_out']


    def mod_fn(self):

        encoder_inputs, decoder_inputs \
            = input_transform(self.phs['encoder_inputs'],
                              self.phs['labels'],
                              self.hps.n_links,
                              self.hps.n_steps_encoder,
                              self.hps.n_steps_decoder,
                              self.hps.n_output_decoder)

        n_stacked_layers = self.hps.n_stacked_layers  # num of layer stacked in RNN
        # dimension of encoder hidden/cell state
        n_hidden_encoder = self.hps.n_hidden_encoder
        # dimension of decoder hidden/cell state
        n_hidden_decoder = self.hps.n_hidden_decoder
        dropout_rate = self.hps.dropout_rate  # dropout rate in RNN unit
        n_output_decoder = self.hps.n_output_decoder

        # Define weights in the transformation layer of decoder
        self.phs['w_out'] = tf.get_variable('Weights_out',
                                            [n_hidden_decoder, n_output_decoder],
                                            dtype=tf.float32,
                                            initializer=tf.truncated_normal_initializer())

        self.phs['b_out'] = tf.get_variable('Biases_out',
                                            shape=[n_output_decoder],
                                            dtype=tf.float32,
                                            initializer=tf.constant_initializer(0.))
        # print('hps.s_attn_flag:\t%s' % self.hps.s_attn_flag)

        with tf.variable_scope('LSTM_Model'):
            # the implement of encoder
            with tf.variable_scope('Encoder'):
                cells = []

                for i in range(n_stacked_layers):
                    with tf.variable_scope('LSTM_{}'.format(i)):
                        # tf.nn.rnn_cell.LSTMCell

                        # tf.nn.rnn_cell.LSTMCell(name='basic_lstm_cell',
                        # cell = tf.nn.rnn_cell.LSTMCell(name='basic_lstm_cell',
                        cell = tf.nn.rnn_cell.LSTMCell(
                            n_hidden_encoder, forget_bias=1.0, state_is_tuple=True)
                        cell = tf.nn.rnn_cell.DropoutWrapper(
                            cell, output_keep_prob=1.0 - dropout_rate)
                        cells.append(cell)
                encoder_cell = tf.contrib.rnn.MultiRNNCell(cells)
                encoder_outputs, encoder_state=self.general_encoder(encoder_inputs,
                                                                encoder_cell)

                #     attn_weights = 0
                # # Calculate a concatenation of encoder outputs to put attention on.
                # top_states = [tf.reshape(e, [-1, 1, encoder_cell.output_size])
                #             for e in encoder_outputs]
                # attention_states = tf.concat(top_states, 1) #hidden states from t1 to T


            # the implement of decoder
            print('starting decoder----------------------------')
            with tf.variable_scope('Decoder'):
                cells = []
                for i in range(n_stacked_layers):
                    with tf.variable_scope('LSTM_{}'.format(i)):
                        # cell = tf.nn.rnn_cell.LSTMCell(name='basic_lstm_cell',
                        cell = rnn.BasicLSTMCell(
                                                    n_hidden_decoder,
                                                    forget_bias=1.0,
                                                    state_is_tuple=True)
                        cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                                             output_keep_prob=1.0 - dropout_rate)
                        cells.append(cell)
                decoder_cell = tf.contrib.rnn.MultiRNNCell(cells)

                # using FCL over static_attr
                decoder_outputs, states = self.general_decoder(decoder_inputs,
                                                                  encoder_state, # as initial state for decoder
                                                                  decoder_cell,
                                                                  loop_function=self._loop_function,
                                                                  )

            # generate outputs
            with tf.variable_scope('Prediction'):
                preds = [tf.matmul(i, self.phs['w_out']) +
                         self.phs['b_out'] for i in decoder_outputs]
        return preds


    def get_loss(self):
        """MSE loss"""
        # reshape
        n_steps_decoder = self.phs['labels'].get_shape()[1].value
        n_output_decoder = self.phs['labels'].get_shape()[2].value
        labels = tf.transpose(self.phs['labels'], [1, 0, 2])
        labels = tf.reshape(labels, [-1, n_output_decoder])
        labels = tf.split(labels, n_steps_decoder, 0)

        # compute empirical loss
        empirical_loss = 0
        # Extra: we can also get separate error at each future time slot
        # print(self.phs)
        for _y, _Y in zip(self.phs['preds'], labels):
            empirical_loss += tf.reduce_mean(tf.pow(_y - _Y , 2))
        self.phs['empirical_loss'] = empirical_loss
        # print(empirical_loss)
        return empirical_loss

    def get_l2reg_loss(self):
        """l2 reg loss"""
        reg_loss = 0
        for tf_var in tf.trainable_variables():
            if 'kernel:' in tf_var.name or 'bias:' in tf_var.name:
                reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))
        return self.lambda_l2_reg * reg_loss

    def train_op(self):
        # Training optimizer
        with tf.variable_scope('Optimizer'):
            global_step = tf.Variable(
                initial_value=0,
                name="global_step",
                trainable=False,
                collections=[tf.GraphKeys.GLOBAL_STEP,
                             tf.GraphKeys.GLOBAL_VARIABLES])
            optimizer = tf.contrib.layers.optimize_loss(
                loss=self.phs['loss'],
                learning_rate=self.hps.learning_rate,
                global_step=global_step,
                optimizer="Adam",
                clip_gradients=self.hps.gc_rate)
        return optimizer

    def summary(self):
        tf.summary.scalar("loss", self.phs['loss'])
        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)
        return tf.summary.merge_all()


class BiLSTM_Model(BaseModel):
    # """docstring for BiLSTM_Model"""
    def __init__(self, hps, mode='train'):
        super(BiLSTM_Model, self).__init__(hps, mode)
        preds=self.mod_fn()
        self.phs['preds'] = preds
        self.phs['loss'] = self.get_loss()  # see at eq.[11]
        tf.add_to_collection('loss', self.phs['loss'])
        self.phs['train_op'] = self.train_op()
        self.phs['summary'] = self.summary()

    def my_self_attention(self,
                          encoder_inputs,
                          grid_features,
                          attention_states,
                          cell,
                          grid_flag=0,
                          output_size=None,
                          dtype=dtypes.float32,
                          scope=None
                    ):
        """self attention in BiLSTM_Model
            @param encoder_inputs: encoder_inputs - the inputs of self attention,
                i.e., a list of 2d tensor with shape of [batch_size, n_links]
            @param attention_states: global_attention_states-
                4D tensor [batch_size, n_links, n_steps_encoder]

            @return: A tuple of form (outputs, state), where:
        """

        # check inputs
        if not encoder_inputs:
            raise ValueError(
                "Must provide at least 1 input to attention encoder.")
        if output_size is None:
            output_size=cell.output_size
        if attention_states.get_shape()[0:2].is_fully_defined():
            raise ValueError('Shape[1] and Shape[2] of attention_states must be know: %s'\
                              % attention_states.get_shape())
        batch_size=array_ops.shape(encoder_inputs[0])[0]


        with vs.variable_scope('self_attn'):
            s_attn_length=attention_states.get_shape()[1].value
            s_n_inputs=attention_states.get_shape()[2].value
            s_attn_size=attention_states.get_shape()[3].value

            # print('attention_states.shape: \t%s', attention_states.shape)
            # print('s_attn_length: \t%s' % s_attn_length)
            # print('s_n_inputs: \t%s' % s_n_inputs)
            # print('s_attn_size: \t%s' % s_attn_size)

            s_hidden=array_ops.reshape(attention_states,
                                        [-1, s_attn_length,  s_n_inputs, s_attn_size])

            s_attn_vec_size=s_attn_size
            self_k=vs.get_variable('AttnUs',
                                        [1, s_n_inputs, s_attn_size,s_attn_vec_size])

            s_hidden_feature=nn_ops.conv2d(
                s_hidden, self_k, [1,1,1,1], "SAME")   # U_s * y^l

            self_v=vs.get_variable(
                "AttenVs", [s_attn_vec_size])
            # print('++++++++++++++++++++++++++++self_v')
            # print(self_v)
            batch_attn_size=array_ops.stack(
                [batch_size, s_attn_length])
            s_attn_fw=array_ops.zeros(batch_attn_size, dtype=dtype)
            s_attn_bw=array_ops.zeros(batch_attn_size, dtype=dtype)

            def self_attention_weight(state): # state: query
                    """
                    """
                    if nest.is_sequence(state):
                        state_list=nest.flatten(state)
                        for q in state_list:
                            ndims=q.get_shape().ndims
                            if ndims:
                                assert ndims==2
                        state=array_ops.concat(state_list, 1)
                    # print("**************state:\t" % tf.shape(state))

                    with tf.variable_scope("AttnWs"):
                        # linear map
                        y=Linear(state, s_attn_vec_size, True)
                        y=array_ops.reshape(
                            y, [-1,1,1,s_attn_vec_size])

                        # Attention mask is a softmax of v_s^{}*tanh(...)
                        s=math_ops.reduce_sum(#g_l^t
                            self_v * math_ops.tanh(s_hidden_feature+y), [2,3])
                        # Sometimes it's not easy to find a measurement to denote similarity between sensors,
                        # here we omit such prior knowledge in eq.[4].
                        # You can use "a = nn_ops.softmax((1-lambda)*s + lambda*sim)" to encode similarity info,
                        # where:
                        #     sim: a vector with length n_sensors, describing the sim between the target sensor and the others
                        #     lambda: a trade-off.
                        a = nn_ops.softmax(s)
                        # a = nn_ops.softmax((1 - lambda) * s + lambda * sim)
                    return a
        # how to get initial state
        initial_state_size=array_ops.stack([batch_size, output_size])
        initial_state_one=[array_ops.zeros(
            initial_state_size, dtype=dtype) for _ in xrange(2)]
        initial_state=[
            initial_state_one for _ in range(len(cell._cells))]

        output_state_fw=initial_state
        output_state_bw=initial_state


        outputs_bw=[]
        outputs_fw=[]
        outputs=[]
        state=[]
        s_attn_weights_fw=[]
        s_attn_weights_bw=[]
        i=0 # time slot index
        # print(len(encoder_inputs))
        # forward
        print('---------------------forward')
        for s_inp in zip(encoder_inputs):
            if i>0:
                vs.get_variable_scope().reuse_variables()

            # print('segmentline--------------------------')
            # print(s_attn_fw.shape) #64, 617
            # print(s_inp[0].get_shape()) # 10, 617
            # print('segmentline--------------------------')
            s_weight = tf.reshape(s_attn_fw[i], [-1, output_size], 's_weight')
            S_weight = tf.reshape(tf.tile(s_weight, [batch_size, 1]), [-1, output_size])
            grid_feature = grid_features[:, i, :, :, :]
            grid_dense = grid_spatial_cnn_model(grid_feature, 64)

            if grid_flag:
                self_x= tf.concat([s_inp[0], S_weight, grid_dense], 1) #self attention vector
            else:
                self_x= tf.concat([s_inp[0], S_weight], 1) #self attention vector

            cell_output_fw, output_state_fw=cell(self_x, output_state_fw)# hidde state, cell state


            with tf.variable_scope('self_attn'):
                s_attn_fw=self_attention_weight(output_state_fw)
            s_attn_weights_fw.append(s_attn_fw)

            with vs.variable_scope("AttentionOutput_fw_Projection"):
                output=cell_output_fw
            outputs_fw.append(output)
            i+=1
        # backward
        print('---------------------backward')
        # print(encoder_inputs)
        for j in  range(len(encoder_inputs)):
            s_b_inp=encoder_inputs[len(encoder_inputs)-j-1]
            # Attention output projection
            if j ==len(encoder_inputs)-1:
                vs.get_variable_scope().reuse_variables()

            # self_x= s_attn_bw * s_b_inp[0]
            s_weight = tf.reshape(s_attn_bw[i], [-1, output_size], 's_weight')
            S_weight = tf.reshape(tf.tile(s_weight, [batch_size, 1]), [-1, output_size])
            # self_x= tf.concat([encoder_inputs[j], S_weight], 1) #self attention vector
            grid_feature = grid_features[:, j, :, :, :]
            grid_dense = grid_spatial_cnn_model(grid_feature, 64)

            if grid_flag:
                self_x= tf.concat([encoder_inputs[j], S_weight, grid_dense], 1) #self attention vector
            else:
                self_x= tf.concat([encoder_inputs[j], S_weight], 1) #self attention vector

            cell_output_bw, output_state_bw=cell(self_x, output_state_bw)# hidde state, cell state

            with tf.variable_scope('self_attn'):
                s_attn_bw=self_attention_weight(output_state_bw)
            #                 print(s_attn_bw.get_shape())
            s_attn_weights_bw.append(s_attn_bw)

            with vs.variable_scope("AttentionOutput_bw_Projection"):
                output=cell_output_bw
            outputs_bw.append(output)
            j+=1

        # combine fw and bw
        if isinstance(output_state_fw, rnn.LSTMStateTuple):  # LstmCell
            state_c = tf.concat(
                (output_state_fw.c, output_state_bw.c), 1, name="bidirectional_concat_c")
            state_h = tf.concat(
                (output_state_fw.h, output_state_bw.h), 1, name="bidirectional_concat_h")
            state = rnn.LSTMStateTuple(c=state_c, h=state_h)
        elif isinstance(output_state_fw, tuple) \
                and isinstance(output_state_fw[0], rnn.LSTMStateTuple):  # MultiLstmCell
            state = tuple(map(
                lambda fw_state, bw_state: rnn.LSTMStateTuple(
                        c=tf.concat((fw_state.c, bw_state.c), 1,
                                name="bidirectional_concat_c"),
                    h=tf.concat((fw_state.h, bw_state.h), 1,
                                name="bidirectional_concat_h")),
                                output_state_fw, output_state_bw))

        else:
            state = tf.concat((output_state_fw, output_state_bw), 1, name="bidirectional_state_concat")

        for k in range(len(encoder_inputs)):
            outputs.append(tf.concat((outputs_fw[k], outputs_bw[len(encoder_inputs)-k-1]), 1))


        return outputs, state, s_attn_weights_bw#         return outputs, state, s_attn_weights

    def general_encoder(self,
                        encoder_inputs,
                        cell,
                        output_size=None,
                        dtype=dtypes.float32,
                        scope=None
                        ):
        # check inputs
        if not encoder_inputs:
            raise ValueError(
                "Must provide at least 1 input to attention encoder.")
        if output_size is None:
            output_size=cell.output_size

        batch_size=array_ops.shape(encoder_inputs[0])[0]

        # how to get initial state
        initial_state_size=array_ops.stack([batch_size, output_size])
        initial_state_one=[array_ops.zeros(
            initial_state_size, dtype=dtype) for _ in xrange(2)]
        initial_state=[
            initial_state_one for _ in range(len(cell._cells))]
        state=initial_state
        initial_state_bw = state
        initial_state_fw = state
        outputs=[]
        cell_fw = cell
        cell_bw = cell
        # print(len(encoder_inputs))
        # print(encoder_inputs[0].get_shape())
        # for s_inp in encoder_inputs:
        #     s_inp = tf.expand_dims(s_inp, axis=1)
        i = 0
        for t in encoder_inputs:
            tmp= tf.expand_dims(t, axis = 1)
            if i == 0:
                inputs = tmp
            else:
                inputs = tf.concat([inputs, tmp], axis = 1 )
            print(inputs.get_shape())
            i +=1
        # inputs  = tf.concat([tf.expand_dims(t, axis = 1) for t in encoder_inputs], axis = 1)

        # print('reshape inputs:\t', print(inputs.get_shape()))

        outputs, state= tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                        cell_bw,
                                                        inputs,
                                                        dtype=dtype)
            # cell_output, state=cell(s_inp[0], state)#
            #output projection

        #     outputs=tf.concat(outputs, 2)

        # combine fw and bw
        output_state_fw = state[0]
        output_state_bw = state[1]

        if isinstance(output_state_fw, rnn.LSTMStateTuple):  # LstmCell
            state_c = tf.concat(
                (output_state_fw.c, output_state_bw.c), 1, name="bidirectional_concat_c")
            state_h = tf.concat(
                (output_state_fw.h, output_state_bw.h), 1, name="bidirectional_concat_h")
            state = rnn.LSTMStateTuple(c=state_c, h=state_h)
        elif isinstance(output_state_fw, tuple) \
                and isinstance(output_state_fw[0], rnn.LSTMStateTuple):  # MultiLstmCell
            state = tuple(map(
                lambda fw_state, bw_state: rnn.LSTMStateTuple(
                        c=tf.concat((fw_state.c, bw_state.c), 1,
                                name="bidirectional_concat_c"),
                    h=tf.concat((fw_state.h, bw_state.h), 1,
                                name="bidirectional_concat_h")),
                                output_state_fw, output_state_bw))

        else:
            state = tf.concat((output_state_fw, output_state_bw), 1, name="bidirectional_state_concat")

        # for k in range(len(encoder_inputs)):
        #     outputs.append(tf.concat((outputs_fw[k], outputs_bw[len(encoder_inputs)-k-1]), 1))
        with vs.variable_scope("OutputProjection"):
            outputs=outputs

        return outputs, state

    def temporal_attention(self,
                           decoder_inputs,
                           grid_features,
                           external_inputs,
                           s_attn_weights,
                           initial_state,
                           attention_states,
                           cell,
                           static_attr_dense,
                           loop_function=None,
                           external_flag=0,
                           s_attn_flag = 0,
                           t_attn_flag=0,
                           grid_flag=0,
                           static_attr_flag=0,
                           output_size=None,
                           dtype=tf.float32,
                           scope=None,
                           initial_state_attention=False,
                           ):
        """ Temporal attention in BiLSTM_Model
        """

        # print('temporal_attention<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        # if self.hps.s_attn_flag=='1':
        #     print('s_attn_flag')

        # if not decoder_inputs:
        #     raise ValueError(
        #         "Please give at leat 1 input to temporal attention decoder.")
        # if not external_inputs:
        #     raise ValueError(
        #         "Please give at least 1 ext_input to temporal attention decoder.")

        if output_size is None:
            output_size=cell.output_size

        # implement of temporal attention
        with vs.variable_scope(
            scope or "temporal_attn", dtype=dtype) as scope:
            dtype=scope.dtype

            batch_size=array_ops.shape(decoder_inputs[0])[0]
            t_attn_length=attention_states.get_shape()[1].value
            if t_attn_length is None:
                t_attn_length=array_ops.shape(attention_states)[1]
            t_attn_size=attention_states.get_shape()[2].value
            hidden=array_ops.reshape(attention_states,
                                    [-1, t_attn_length, 1, t_attn_size])
            # Size of state vector for attention
            t_attn_vec_size=t_attn_size
            w=vs.get_variable(
                "Attn_Wd", [1,2,t_attn_size, t_attn_vec_size]) # W_d
            hidden_feature=nn_ops.conv2d(
                hidden, w, [1, 1,1,1], "SAME") #W_d * h_o
            v=vs.get_variable("Attn_v",[t_attn_vec_size]) # v_d
            state = initial_state

            def temporal_attention_weight(query):

                """
                    Put attention masks on self_hidden using self_hidden_features and query.
                """

                if nest.is_sequence(query):
                    query_list=nest.flatten(query)

                    for q in query_list:

                        ndims=q.get_shape().ndims
                        if ndims:
                            assert ndims==2
                    query = array_ops.concat(query_list, 1)


                # print('*************8***********state:\t%s' %  tf.shape(state[0]))
                with vs.variable_scope("Attn_Wpd"):
                    # Linear map
                    y=Linear(query, t_attn_vec_size, True)
                    y=array_ops.reshape(
                        y, [-1,1,1,t_attn_vec_size])

                    # Attention mask is a softmax of v_d^{\top} * tanH(...)
                    s=math_ops.reduce_sum(v*math_ops.tanh(hidden_feature + y + 1e-10),
                                           [2, 3])

                    # cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))

                    a=nn_ops.softmax(s)
                    d=math_ops.reduce_sum(
                        array_ops.reshape(a, [-1,t_attn_length, 1, 1])* hidden, [1,2])
                return array_ops.reshape(d, [-1, t_attn_size])

            if initial_state_attention:
                t_attn=temporal_attention_weight(inital_state) # temporal attention vector
            else:
                batch_attn_size=array_ops.stack([batch_size, t_attn_size])
                t_attn=array_ops.zeros(batch_attn_size, dtype=dtype)
                t_attn.set_shape([None, t_attn_size])



            # decoder for labels
            i=0
            outputs=[]
            prev=None

            print('==========================decoder_inputs')
            print(decoder_inputs)
            for inp, ext_inp in zip(decoder_inputs, external_inputs):
                # print(i)
                if i>0:
                    vs.get_variable_scope().reuse_variables()
                # If loop_function is set, we use it instead of decoder_inputs.
                if loop_function is not None and prev is not None:
                    # with vs.get_variable_scope("loop_function", reuse=True):
                    with vs.variable_scope("loop_function", reuse=True):

                        inp=loop_function(prev, i)

                # Merge input and previous attentions into one vector of the right size.
                input_size = inp.get_shape().with_rank(2)[1]
                if input_size.value is None:
                    raise ValueError(
                        "Could not infer input size from input: %s" % inp.name)


                if external_flag and t_attn_flag:
                    x = Linear([inp] + [ext_inp] + [t_attn], input_size, True)
                elif t_attn_flag:
                    x = Linear([inp] + [t_attn], input_size, True)
                else:
                    x = Linear([inp], input_size, True)

                # # Run the RNN.
                cell_output, state = cell(x, state) # query: cell state

                if i == 0 and initial_state_attention:
                    with vs.variable_scope(vs.get_variable_scope(), reuse=True):

                        t_attn = temporal_attention_weight(state)
                else:
                    t_attn = temporal_attention_weight(state)

                with vs.variable_scope("AttnOutputProjection"):
                    if t_attn_flag and static_attr_flag:
                        fclInput = tf.concat([cell_output, t_attn, static_attr_dense], 1)
                    elif t_attn_flag:
                        fclInput = tf.concat([cell_output, t_attn], 1)
                    else:
                        fclInput = cell_output
                    output = tf.contrib.layers.fully_connected(fclInput, output_size, activation_fn=tf.nn.relu)

                if loop_function is not None:
                    prev = output
                outputs.append(output)
                i += 1
        return outputs, state

    def _loop_function(self, prev, _):
        """loop function used in the decoder to generate the next inupt"""
        return tf.matmul(prev, self.phs['w_out']) + self.phs['b_out']


    def mod_fn(self):
        encoder_attention_states, \
        encoder_inputs, decoder_inputs,\
        encoder_external_inputs, decoder_external_inputs,\
        grid_features, static_attr \
            = input_transform(self.phs['encoder_inputs'],
                              self.phs['labels'], # decoder_inputs
                              self.phs['self_attn_states'],
                              self.phs['encoder_external_inputs'],
                              self.phs['decoder_external_inputs'],
                              self.phs['grid_features'],
                              self.phs['static_attr'])


        n_stacked_layers = self.hps.n_stacked_layers  # num of layer stacked in RNN
        # dimension of encoder hidden/cell state
        n_hidden_encoder = self.hps.n_hidden_encoder
        # dimension of decoder hidden/cell state
        n_hidden_decoder = self.hps.n_hidden_decoder
        dropout_rate = self.hps.dropout_rate  # dropout rate in RNN unit
        n_output_decoder = self.hps.n_output_decoder

        # Define weights in the transformation layer of decoder
        self.phs['w_out'] = tf.get_variable('Weights_out',
                                            [n_hidden_decoder, n_output_decoder],
                                            dtype=tf.float32,
                                            initializer=tf.truncated_normal_initializer())

        self.phs['b_out'] = tf.get_variable('Biases_out',
                                            shape=[n_output_decoder],
                                            dtype=tf.float32,
                                            initializer=tf.constant_initializer(0.))
        # print('hps.s_attn_flag:\t%s' % self.hps.s_attn_flag)

        with tf.variable_scope('BiLSTM_Model'):
            # the implement of encoder
            with tf.variable_scope('Encoder'):
                cells = []

                for i in range(n_stacked_layers):
                    with tf.variable_scope('LSTM_{}'.format(i)):
                        cell = rnn.BasicLSTMCell(
                            n_hidden_encoder, forget_bias=1.0, state_is_tuple=True)
                        cell = tf.nn.rnn_cell.DropoutWrapper(
                            cell, output_keep_prob=1.0 - dropout_rate)
                        cells.append(cell)
                encoder_cell = tf.contrib.rnn.MultiRNNCell(cells)

                if self.hps.s_attn_flag:

                    encoder_outputs, encoder_state, attn_weights = self.my_self_attention(encoder_inputs,
                                                                                          grid_features,
                                                                                          encoder_attention_states,
                                                                                          encoder_cell,
                                                                                          grid_flag=self.hps.grid_flag)
                    print('s_attn_flag:\t%s' % self.hps.s_attn_flag)
                    print(len(encoder_outputs))
                    print(encoder_outputs[0].get_shape()) # (?, 646)
                    top_states = [tf.reshape(e, [-1, 1, 2*encoder_cell.output_size])
                            for e in encoder_outputs]
                    attention_states = tf.concat(top_states, 1) #hidden states from t1 to T (?, time_lag, size): (?, 6, 646)
                    # print(encoder_state[0].get_shape())
                else:
                    print('s_attn_flag:\t%s' % 'no')
                    encoder_outputs, encoder_state=self.general_encoder(encoder_inputs,
                                                                   encoder_cell)
                    print(encoder_outputs[0].get_shape())

                    encoder_outputs = tf.concat([encoder_outputs[0], encoder_outputs[1]], axis = 2) # concat final fw hidden state and final bw state

                    print('prepare for temporal_attention')
                    attention_states = tf.concat(encoder_outputs, 1) #hidden states from t1 to T
                    print('attention_states.shape')
            print(encoder_state)
            print(attention_states.get_shape())
            # the implement of decoder
            with tf.variable_scope('Decoder'):
                cells = []
                for i in range(n_stacked_layers):
                    with tf.variable_scope('LSTM_{}'.format(i)):
                        cell = rnn.BasicLSTMCell(n_hidden_decoder,
                                                 forget_bias=1.0,
                                                 state_is_tuple=True)
                        cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                                             output_keep_prob=1.0 - dropout_rate)
                        cells.append(cell)
                decoder_cell = tf.contrib.rnn.MultiRNNCell(cells)

                # using FCL over static_attr
                static_attr_output_size = self.hps.n_static_attr_output_size
                static_attr_dense = static_attr_fcl_model(static_attr, static_attr_output_size)
                decoder_outputs, states = self.temporal_attention(decoder_inputs,
                                                                  grid_features,
                                                                  decoder_external_inputs,
                                                                  attn_weights,
                                                                  encoder_state,
                                                                  attention_states,
                                                                  decoder_cell,
                                                                  static_attr_dense,
                                                                  loop_function=self._loop_function,
                                                                  s_attn_flag=self.hps.s_attn_flag,
                                                                  external_flag=self.hps.external_flag,
                                                                  t_attn_flag=self.hps.t_attn_flag,
                                                                  grid_flag=self.hps.grid_flag,
                                                                  static_attr_flag=self.hps.static_attr_flag)


                # print('...........................decoder_outputs')
                # print(decoder_outputs)
                # print('...........................decoder_state')
                # print(states)
            # generate outputs
            with tf.variable_scope('Prediction'):
                preds = [tf.matmul(i, self.phs['w_out']) +
                         self.phs['b_out'] for i in decoder_outputs]

        return preds

    def get_loss(self):
        """MSE loss"""
        # reshape
        n_steps_decoder = self.phs['labels'].get_shape()[1].value
        n_output_decoder = self.phs['labels'].get_shape()[2].value
        labels = tf.transpose(self.phs['labels'], [1, 0, 2])
        labels = tf.reshape(labels, [-1, n_output_decoder])
        labels = tf.split(labels, n_steps_decoder, 0)

        # compute empirical loss
        empirical_loss = 0
        # Extra: we can also get separate error at each future time slot
        # print(self.phs)
        for _y, _Y in zip(self.phs['preds'], labels):
            empirical_loss += tf.reduce_mean(tf.pow(_y - _Y , 2))
        self.phs['empirical_loss'] = empirical_loss
        # print(empirical_loss)
        return empirical_loss

    def get_l2reg_loss(self):
        """l2 reg loss"""
        reg_loss = 0
        for tf_var in tf.trainable_variables():
            if 'kernel:' in tf_var.name or 'bias:' in tf_var.name:
                reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))
        return self.lambda_l2_reg * reg_loss

    def train_op(self):
        # Training optimizer
        with tf.variable_scope('Optimizer'):
            global_step = tf.Variable(
                initial_value=0,
                name="global_step",
                trainable=False,
                collections=[tf.GraphKeys.GLOBAL_STEP,
                             tf.GraphKeys.GLOBAL_VARIABLES])
            optimizer = tf.contrib.layers.optimize_loss(
                loss=self.phs['loss'],
                learning_rate=self.hps.learning_rate,
                global_step=global_step,
                optimizer="Adam",
                clip_gradients=self.hps.gc_rate)
        return optimizer

    def summary(self):
        tf.summary.scalar("loss", self.phs['loss'])
        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)
        return tf.summary.merge_all()

# encoder_inputs: #[-1, number of links]
# encoder_inputs: (number of links, batch_size, 1)
# Global inp: (batch_size, 1)


import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest
import numpy as np
import pandas as pd
import h5py
import os


def load_data(path):
    train_data = np.load(path + 'train.npz')
    valid_data = np.load(path + 'valid.npz')
    test_data = np.load(path + 'test.npz')

    train_data_list = [train_data['x'], train_data['y']]
    valid_data_list = [valid_data['x'], valid_data['y']]
    test_data_list = [test_data['x'], test_data['y']]

    print('train x shape:\t', train_data['x'].shape)
    print('train y shape:\t', train_data['y'].shape)


    print('valid x shape:\t', valid_data['x'].shape)
    print('valid y shape:\t', valid_data['y'].shape)


    print('test x shape:\t', test_data['x'].shape)
    print('test y shape:\t', test_data['y'].shape)

    # train x shape:   (267008, 1344, 2)
    # train y shape:   (267008, 336, 2)

    return train_data_list, valid_data_list, test_data_list


def get_train_valid_test(data, batch_size, num_all_days, n_links, train_r=0.8, valid_r=0.1, test_r=0.1):
    shape =data[0].shape
    x = num_all_days
    print('------------------x:\t', x)
    print('------------------data:\t', shape)
    # print(data[0].shape)
    train_data, valid_data, test_data = [[]]*len(data), [[]]*len(data), [[]]*len(data)
    for i in range(len(data)):
        num_train = int(x * train_r) * n_links
        num_train_valid =int(x * (train_r + valid_r)) * n_links
        num_all = x * n_links
        ind_train = range(num_train)
        ind_valid = range(num_train, num_train_valid)
        ind_test = range(num_train_valid, num_all)

        train_data[i] = data[i][ind_train, :, :]
        valid_data[i] = data[i][ind_valid, :, :]
        test_data[i] = data[i][ind_test, :, :]
    print('train_data shape:\t', train_data[0].shape)
    print('valid_data shape:\t', valid_data[0].shape)
    print('test_data shape:\t', test_data[0].shape)
    return train_data, valid_data, test_data


def get_lstm_flow_data(region, data_path, batch_size, p, n_days, steps_of_one_day):
    # reformat traffic flow data from BJ
    speedMatrix = np.load(data_path + 'all_amapHK_speedMatrix.npz')
    speedMatrix = speedMatrix['x']
    input_scale_1, input_scale_7, labels = [], [[]]*p, []

    if 1:
    # for k in range(len(data_1)):
    #     speedMatrix = np.array(data_1[k])

        print('original size:\t', speedMatrix.shape)
        speedMatrix = np.transpose(speedMatrix)
        speedMatrix = np.expand_dims(speedMatrix, axis=-1)
        (x, y, z) = speedMatrix.shape
        print("x, y, z", x, y, z)
        print(steps_of_one_day*int(x / steps_of_one_day))
        speedMatrix = np.array(speedMatrix[:steps_of_one_day*int(x / steps_of_one_day), :, :])
        print(speedMatrix.shape)
        (x, y, z) = speedMatrix.shape
        speedMatrix = np.transpose(speedMatrix, [1, 0, 2])
        print(speedMatrix.shape)

        # speedMatrix = np.array(speedMatrix.tolist())

        speedMatrix = np.expand_dims(speedMatrix, axis=1)
        print('speedMatrix size:\t', speedMatrix.shape) #(5000, 1, 1488, 1)
        speedMatrix = np.concatenate(np.split(speedMatrix, int(x / steps_of_one_day), axis=2), axis = 1) # split by day
        print('size after split:\t', speedMatrix.shape) #(1024, 101, 48, 2)
        num_all_days =  speedMatrix.shape[1] - p - n_days
        n_links = speedMatrix.shape[0]
        speedMatrix = np.transpose(speedMatrix, [3, 0, 1, 2]) # 2, 1024, 101, 48
        print('speedMatrix size before sperating:\t', speedMatrix.shape) # (1, 5000, 31, 720)
        speedMatrix = speedMatrix[:, :, :, 270:570]
        print('speedMatrix size before sperating:\t', speedMatrix.shape) # (1, 5000, 31, 720)
        used_steps_of_one_day = speedMatrix.shape[-1]
        # traing input

        for i in range(n_days, int(x / steps_of_one_day)-p):
            # print(speedMatrix[i - n_days: i][0].type)
            tmp_sam = speedMatrix[:, :, i - n_days: i, :]

            tmp_sam = np.reshape(tmp_sam, [tmp_sam.shape[0], tmp_sam.shape[1], n_days * used_steps_of_one_day])
            # print('tmp_samshape:\t', tmp_sam.shape) # (2, 1, 1024, 1344)
            tmp_sam_label = speedMatrix[:, :, i : i+p, :]
            tmp_sam_label = np.reshape(tmp_sam_label, [tmp_sam_label.shape[0], tmp_sam_label.shape[1], p * used_steps_of_one_day])
            if input_scale_1 == []:
                input_scale_1 = tmp_sam
            else:
                input_scale_1 = np.concatenate([input_scale_1, tmp_sam], axis = 1)

            if labels == []:
                labels = tmp_sam_label
            else:
                labels = np.concatenate([labels, tmp_sam_label], axis = 1)

    input_scale_1 = np.asarray(input_scale_1)
    labels = np.asarray(labels)


    print('input_scale_1:\t', input_scale_1.shape)
    input_scale_1 = np.transpose(input_scale_1, [1, 2, 0])
    labels = np.transpose(labels, [1, 2, 0])


    # input_scale_1 = np.expand_dims(input_scale_1, axis=-1)
    # labels = np.expand_dims(labels, axis=-1)
    print('input_scale_1:\t', input_scale_1.shape)
    print('labels:\t', labels.shape)
    print(input_scale_1.max())
    print(input_scale_1.min())
    data = [input_scale_1, labels]
    # """
    train_data, valid_data, test_data = get_train_valid_test(data, batch_size, num_all_days, n_links)
    [x_train, y_train] = train_data
    [x_valid, y_valid] = valid_data
    [x_test, y_test] = test_data

    output_dir = '/public/hezhix/DataParse/DurationPre/Data/unScale/%s/lstm2lstm_all_amapHK/' % region
    mkdir_file(output_dir)
    for cat in ["train", "valid", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(output_dir, "%s.npz" % cat),
            x=_x,
            y=_y)

    return train_data, valid_data, test_data


def load_taxiBJ_data(batch_size, region):

    data_1 = [[]] *4
    for i in range(13, 17): # from year 2013 to 2016
        file_path = '/public/hezhix/DataParse/DurationPre/dnn/TrafficPre/DeepST/data/TaxiBJ/BJ%d_M32x32_T30_InOut.h5' % i
        f = h5py.File(file_path)
        j = 0
        for ke in f.keys():
            if j == 0:
                temp = f.get(ke).value
                temp = temp.reshape(temp.shape[0], temp.shape[1], int(temp.shape[2]) * int(temp.shape[3]))
                temp = temp.transpose([0, 2, 1])
                data_1[i-13] = temp
                # data_1[i-13] = tp_dset.read_direct(arr)
                print(temp.shape)
            else:
                pass
            j +=1
            print('j:\t', j)
    max_val = np.max([data_1[0].max(), data_1[1].max(), data_1[2].max(), data_1[3].max()])
    print(max_val)
    data_1  = data_1 / max_val
    p = 7
    n_days = 28
    steps_of_one_day  = 48
    print('data:\t', data_1[0].shape)
    train_data,valid_data, test_data = get_lstm_flow_data(region, data_1, batch_size, p, n_days, steps_of_one_day)
    print('train_data shape:\t', train_data[0].shape)
    print('valid_data shape:\t', valid_data[0].shape)
    print('test_data shape:\t', test_data[0].shape)
    # print(train_data[0])
    return train_data,valid_data, test_data


def load_taxiBJ_data_split(batch_size, region, split_id):
    data_1 = [[[]]*4] *4
    max_val = 1292
    for i in range(13, 17):
        file_path = '/public/hezhix/DataParse/DurationPre/dnn/TrafficPre/DeepST/data/TaxiBJ/BJ%d_M32x32_T30_InOut.h5' % i
        f = h5py.File(file_path)
        j = 0
        for ke in f.keys():
            if j == 0:
                temp = f.get(ke).value
                temp = np.array(temp.tolist())
                temp = temp / max_val
                temp_split_1 = np.split(temp, 2, axis = 2)
                k = 0
                for temp_split in temp_split_1:
                    temp_split_2 = np.split(temp_split, 2, axis = 3)
                    for temp_split_3 in temp_split_2:
                        print(temp_split_3.shape)
                        temp_split_3 = temp_split_3.reshape(temp_split_3.shape[0], temp_split_3.shape[1], int(temp_split_3.shape[2]) * int(temp_split_3.shape[3]))
                        temp_split_3 = np.transpose(temp_split_3, [0, 2, 1])
                        print(temp_split_3.shape)
                        print('\n')
                        data_1[k][i-13] = temp_split_3
                        k += 1
                # print(temp.shape)
            else:
                pass
            j +=1
            print('j:\t', j)
    split_data = [[]] * 4
    for ii in range(len(data_1)):
        split_data[ii] = data_1[ii]
    data_need  = split_data[split_id]
    # max_val = np.max([data_need[0].max(), data_need[1].max(), data_need[2].max(), data_need[3].max()])
    # print(max_val)
    # data_need  = data_need / max_val
    pre = 7
    n_days = 28
    steps_of_one_day = 48
    region = region + '%s' % split_id
    train_data,valid_data, test_data = get_lstm_flow_data(region, data_need, batch_size, pre, n_days, steps_of_one_day)



def load_bikeNYC_data(batch_size, region):

    file_path = '/public/hezhix/DataParse/DurationPre/dnn/TrafficPre/DeepST/data/BikeNYC/NYC14_M16x8_T60_NewEnd.h5'
    #
    f = h5py.File(file_path)
    j = 0
    for ke in f.keys():
        if j == 0:
            temp = f.get(ke).value
            temp = temp.reshape(temp.shape[0], temp.shape[1], int(temp.shape[2]) * int(temp.shape[3]))
            temp = temp.transpose([0, 2, 1])
            data_1= temp
            print(temp.shape)
        else:
            pass
        j +=1
        print('j:\t', j)
    print(data_1)
    print(data_1.shape)
    data_2  = np.array(data_1.tolist()) # for maximun value
    max_val = np.max(data_2)
    print(max_val)
    data_1  = data_1 / max_val
    data = [[]]
    data[0] = data_1
    pre = 7
    n_days = 28
    steps_of_one_day = 24
    train_data,valid_data, test_data = get_lstm_flow_data(region, data, batch_size, pre, n_days, steps_of_one_day)

    # print(train_data[0])
    return train_data,valid_data, test_data

def Linear(args,
           output_size,
           bias,
           bias_initializer=None,
           kernel_initializer=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_initializer: starting value to initialize the bias
        (default is all zeros).
      kernel_initializer: starting value to initialize the weight.
    Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    # print('***************args:\t%s' % args)
    shapes = [a.get_shape() for a in args]
    for shape in shapes:
        if shape.ndims != 2:
            raise ValueError(
                "linear is expecting 2D arguments: %s" % shapes)
        if shape[1].value is None:
            raise ValueError("linear expects shape[1] to be provided for shape %s, "
                             "but saw %s" % (shape, shape[1]))
        else:
            total_arg_size += shape[1].value

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    scope = vs.get_variable_scope()
    # print('*******************scope:\t%s' % scope.original_name_scope)
    # weights = tf.Variable([total_arg_size, output_size],
    #                       dtype=tf.float32,
    #                       name='kernel',
    #                       initializer=kernel_initializer)
    with vs.variable_scope(scope) as outer_scope:
    # with vs.variable_scope(scope, reuse=tf.AUTO_REUSE) as outer_scope:
        # print('*************outer_scope:\t%s' % outer_scope.original_name_scope)
        weights = vs.get_variable(
            "kernel",
            [total_arg_size, output_size],
            dtype=dtype,
            initializer=kernel_initializer)

        if len(args) == 1:
            res = math_ops.matmul(args[0], weights)
        else:
            res = math_ops.matmul(array_ops.concat(args, 1), weights)
        if not bias:
            return res
        with vs.variable_scope(outer_scope) as inner_scope:
            inner_scope.set_partitioner(None)
            if bias_initializer is None:
                bias_initializer = init_ops.constant_initializer(
                    0.0, dtype=dtype)
            biases = vs.get_variable(
                "bias", [output_size],
                dtype=dtype,
                initializer=bias_initializer)
        return nn_ops.bias_add(res, biases)

def load_link_static_attr_pickle_to_df(location):
    file_path = '/public/hezhix/DataParse/DurationPre/Data/Scale/HongKong_721/line_link_static_attr_df_%s.npy' % location
    if location == 'Los':
        file_path = '/public/hezhix/DataParse/DurationPre/Data/Scale/%s_721/line_link_static_attr_df_%s.npy' % (location, location)

    link_static_attr_np = np.load(file_path)

    print('static_attr shape:\t', link_static_attr_np.shape)
    return link_static_attr_np

def load_data_seq2seq(input_path, mode, n_steps_encoder, n_steps_decoder):

    global_inp_index=np.load(
        input_path+"{}_{}_{}_global_inp_index.npy".format(mode, n_steps_encoder, n_steps_decoder))
    endid = 0 - n_steps_encoder - n_steps_decoder
    global_inp_index = global_inp_index[: endid]
    global_inp_index=global_inp_index.astype(int)
    print(global_inp_index.shape)


    print('load --%s-- data successfully!!!\n' % mode)

    return [global_inp_index]


def load_global_inputs(input_path1, mode, n_steps_encoder, n_steps_decoder):
    global_inputs=np.load(
        input_path1 +"{}_{}_{}_samples_input_data_without_nan.npy".format(mode, n_steps_encoder, n_steps_decoder))

    print('global_inputs.shape:\t', global_inputs.shape)
    # print('-----load global_inputs successfully!!')
    print('load --base-- data successfully!!!\n')
    return global_inputs



def load_encoder_inputs(input_path1, state_path, mode, n_steps_encoder, n_steps_decoder):
    encoder_inputs=np.load(
        input_path1 +"{}_{}_{}_samples_input_data_without_nan.npy".format(mode, n_steps_encoder, n_steps_decoder))

    print('encoder_inputs.shape:', encoder_inputs.shape)
    # print('-----load encoder_inputs successfully!!')
    self_attn_states=np.load(
        state_path + "{}_{}_1_spatial_attn_states.npy".format(mode, n_steps_encoder))
    self_attn_states=self_attn_states[:,:,:,:n_steps_encoder]

    print('self_attn_states.shape:', self_attn_states.shape)# [990,617,1,10]
    print('load --base-- data successfully!!!\n')
    return encoder_inputs, self_attn_states

def load_grid_features(path):
    grid_features = np.load(path)

    return grid_features

def basic_hyperparams():
    return tf.contrib.training.HParams(
        # GPU arguments
        gpu_id='0',

        # model parameters
        learning_rate=1e-3,
        lambda_l2_reg=1e-3,
        gc_rate=2.5,  # to avoid gradient exploding
        dropout_rate=0.3,
        n_stacked_layers=2,
        s_attn_flag=True,
        t_attn_flag=True,
        external_flag=1,
        gru_flag=0,
        grid_flag=1,
        static_attr_flag=1,

        # encoder parameter
        ass_n_links=0,
        n_links=323,
        n_input_encoder=1, # nuber of local series features of each sensors
        n_steps_encoder=10,  # time steps
        n_grid = 8,
        grid_cnn_output_size=64,
        n_hidden_encoder=256,  # size of hidden units

        # decoder parameter
        n_input_decoder=323,
        n_external_input=5,
        n_steps_decoder=1, # predict 1 time slot
        n_hidden_decoder=256,
        n_static_attr_output=64,
        n_output_decoder=2  # size of the decoder output
    )

def count_total_params():
    """ count the parameters in the model """
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(shape)
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        # print(variable_parameters)
        total_parameters += variable_parameters
    print(total_parameters)

def shuffle_data(training_data, n_steps_encoder, n_steps_decoder):
    """ shuffle data"""
    # print('---------------training_data[0].shape[0]:\t%s' % training_data[-1].shape[0])

    shuffle_index = np.random.permutation(training_data[0].shape[0])
    new_training_data = []
    for inp in training_data:
        new_training_data.append(inp[shuffle_index])
    return new_training_data


def mkdir_file(file_path):
    directory = os.path.dirname(file_path)
    if not  os.path.exists(directory):
        os.makedirs(directory)

def get_lstm_flow_data_speed(speedMatrix, batch_size, region, region_dict):
    print('original size:\t', speedMatrix.shape)
    (x, y, z) = speedMatrix.shape
    speedMatrix = np.squeeze(speedMatrix)

    s_to_e  =[0, 202]
    if region in ['HK-KL', 'ST', 'TM']:
        s_to_e = region_dict[region]
    else:
        print('need input rational region name')
        return

    speedMatrix = speedMatrix[s_to_e[0]: s_to_e[1], :]
    speedMatrix = np.transpose(speedMatrix)
    speedMatrix  = speedMatrix/ np.max(speedMatrix)
    # speedMatrix = np.split(speedMatrix, int(z / 144), axis=1) # split by day
    # print('size after split:\t', speedMatrix.shape)

    # traing input
    p = 7
    n_days = 28
    input_scale_1, input_scale_7, labels = [], [[]]*p, []
    for i in range(n_days * 144, len(speedMatrix) - p* 144, 144):
        # print(list(range(i-28, i, 7)))
        # print(slice(i-28, i, 7))
        input_scale_1.append(speedMatrix[i - n_days * 144: i, :])
        # for j in range(config.pre):
        #     input_scale_7[j].append(speedMatrix[slice(i-28+j, i, 7)])
        labels.append(speedMatrix[i: (i + p * 144), :])

    input_scale_1 = np.asarray(input_scale_1)
    labels = np.asarray(labels)
    num_all_days = input_scale_1.shape[1]
    # input_scale_1 = np.expand_dims(input_scale_1, axis=-1)
    # labels = np.expand_dims(labels, axis=-1)
    print('encoder_inputs:\t', input_scale_1.shape)
    print('labels:\t', labels.shape)
    print(input_scale_1.max())
    print(input_scale_1.min())
    data = [input_scale_1, labels]

    train_data, valid_data, test_data = get_train_valid_test(data, batch_size, steps_of_one_day)
    return train_data, valid_data, test_data



def get_batch_input_dict(model, train_data):

    feed_dict = {
        model.phs['encoder_inputs']: train_data[0],
        model.phs['labels']: train_data[1]}
    return feed_dict


# training_data= [mode_local_inp, global_inp_index, global_attn_index, mode_ext_inp, mode_labels]
def get_batch_feed_dict(model, k, batch_size, training_data, global_inputs, n_steps_encoder, n_steps_decoder):
    """ get feed_dict of each batch in a training epoch"""

    train_global_inp_ind = training_data[0]

    batch_global_inp = train_global_inp_ind[k:k + batch_size]

    enc_tmp = []
    for j in batch_global_inp:
        enc_tmp.append(
            global_inputs[j: j + n_steps_encoder, :])
    enc_tmp = np.array(enc_tmp)

    dec_tmp = []
    for j in batch_global_inp:
        dec_tmp.append(
            global_inputs[j + n_steps_encoder:\
             j + n_steps_encoder + n_steps_decoder, :])
    dec_tmp = np.array(dec_tmp)

    # print(enc_tmp.shape)
    # print(dec_tmp.shape)
    # print('\n')
    # # feed_dict
    feed_dict = {model.phs['encoder_inputs']: enc_tmp,
                 model.phs['labels']: dec_tmp}

    return feed_dict

def get_valid_batch_feed_dict(model, valid_indexes, k, valid_data, valid_global_inputs, n_steps_encoder, n_steps_decoder):
    """ get feed_dict of each batch in the validation set"""

    valid_global_inp_ind = valid_data[0]

    batch_global_inp = valid_global_inp_ind[valid_indexes[k]:valid_indexes[k + 1]]

    # get encoder_inputs
    enc_tmp = []
    for j in batch_global_inp:
        enc_tmp.append(
            valid_global_inputs[j: j + n_steps_encoder, :])
    enc_tmp = np.array(enc_tmp)

    # get decoder_inputs
    dec_tmp = []
    for j in batch_global_inp:
        dec_tmp.append(
            valid_global_inputs[j + n_steps_encoder: j + n_steps_encoder + n_steps_decoder, :])
    dec_tmp = np.array(dec_tmp)
    # feed_dict
    feed_dict = {model.phs['encoder_inputs']: enc_tmp,
                 model.phs['labels']: dec_tmp}

    return feed_dict

def override_from_dict(self, values_dict):
    """Override hyperparameter values, parsing new values from a dictionary.
    Args:
      values_dict: Dictionary of name:value pairs.
    Returns:
      The `HParams` instance.
    Raises:
      ValueError: If `values_dict` cannot be parsed.
    """
    for name, value in values_dict.items():
      self.set_hparam(name, value)
    return self



def to_var(var):
    if torch.is_tensor(var):
        var = Variable(var)
        if torch.cuda.is_available():
            var = var.cuda()
        return var
    if isinstance(var, int) or isinstance(var, float):
        return var
    if isinstance(var, dict):
        for key in var:
            var[key] = to_var(var[key])
        return var
    if isinstance(var, list):
        var = map(lambda x: to_var(x), var)
        return var



if __name__ == "__main__":
    region = 'amapHK'
    speed_folder = '/public/hezhix/DataParse/DurationPre/Data/unScale/' \
                        + 'amapHK/speedMatrix/'
    # speedMatrix = np.load(speed_folder + 're_range_amapHK_speedMatrix_without_nan.npy')
    npz_dir = speed_folder + 'npz/'


    """
    mkdir_file(npz_dir)
    np.savez_compressed(
        os.path.join(npz_dir, "amapHK_speedMatrix_without_nan.npz"),
        x=speedMatrix)
    """
    """

    # merge multiple data for months
    data = []
    for i in range(8):
        sub_data = np.load(npz_dir + 'training_duration_%s.npz' % i)
        print(sub_data.files)
        sub_data = sub_data['x']
        print(sub_data.shape) # (5000, 22320) # i = 7: (2886, 22320)
        if data == []:
            data = sub_data
        else:
            data = np.concatenate((data, sub_data), axis=0)

    test_data = np.load(npz_dir + 'testing_duration.npz')
    test_data = test_data['x']
    print('test_data shape:\t', test_data.shape)    # data shape: (37886, 22320) (n_links, len_slots)
    print('data shape:\t', data.shape)    # data shape: (37886, 22320) (n_links, len_slots)
    data = np.concatenate((data, test_data), axis=1)

    print('max:\t', data.max())
    print('min:\t', data.min())
    print('data shape:\t', data.shape)    # data shape: (37886, 22320) (n_links, len_slots)

    # compress in .npz
    mkdir_file(npz_dir)
    np.savez_compressed(
        os.path.join(npz_dir, "all_amapHK_speedMatrix.npz"),
        x=data)
    path = npz_dir
    # train_data = np.load(path + 'all_amapHK_speedMatrix.npz')
    # print(train_data['x'].shape)

    batch_size = 256
    p = 1
    n_days = 1
    steps_of_one_day = 720

    get_lstm_flow_data(region, npz_dir, batch_size, p, n_days, steps_of_one_day)
    """


    PATH = '/public/hezhix/DataParse/DurationPre/Data/unScale/amapHK/lstm2lstm_all_amapHK/old/'


    test_data = np.load(PATH + 'test.npz')
    print(test_data['x'].shape)

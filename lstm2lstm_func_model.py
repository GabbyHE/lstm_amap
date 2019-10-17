import os
import json
import math
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from lstm2lstm_utils import load_data
from lstm2lstm_utils import load_global_inputs
from lstm2lstm_utils import load_link_static_attr_pickle_to_df
from lstm2lstm_utils import basic_hyperparams
from lstm2lstm_utils import get_batch_feed_dict
from lstm2lstm_utils import shuffle_data
from lstm2lstm_utils import get_valid_batch_feed_dict
from lstm2lstm_utils import override_from_dict
from matplotlib import pyplot as plt
from lstm2lstm_Model import LSTM_Model, BiLSTM_Model
from lstm2lstm_utils import load_taxiBJ_data_split, load_bikeNYC_data, get_lstm_flow_data, get_batch_input_dict, load_taxiBJ_data, load_data



def root_mean_squared_error(labels, preds):
    idx = np.nonzero(labels)
    total_size = np.size(labels)
    return np.sqrt(np.sum(np.square(labels - preds)) / total_size)

def mean_absolute_error(labels, preds):
    total_size = np.size(labels)
    return np.sum(np.abs(labels - preds)) / total_size


def mean_absolute_percentage_error(y_true, y_pred):
    idx = np.nonzero(y_true)
    return np.mean(np.abs((y_true[idx] - y_pred[idx]) / y_true[idx])) * 100


def mean_absolute_relative_error(labels, preds):
    idx = np.nonzero(labels)
    total_size = np.size(labels)
    # print(np.sum(np.abs(labels - preds)/labels))
    # print(total_size)
    # print(np.min(labels))
    # return np.sum(np.abs(labels - preds)/labels) / total_size
    return np.mean(np.abs((labels[idx] - preds[idx]) / labels[idx])) * 100

def sy_mean_absolute_relative_error(labels, preds):
    total_size = np.size(labels)
    # print(np.sum(np.abs(labels - preds)/labels))
    # print(total_size)
    # print(np.min(labels))
    # print()
    return np.sum(2*np.abs(labels - preds)/(labels+preds)) / total_size

def mkdir_file(file_path):
    directory = os.path.dirname(file_path)
    if not  os.path.exists(directory):
        os.makedirs(directory)

def load_params(args):
    attn_flag = [[True,True,False,False], [True,False,True,False]]
    Models = ['lstm2lstm']
    pre_data = {'Seattle': [{'step_size': 5, 'n_links': 323, 'speed_max': 100}],
                'HongKong': [{'step_size': 10, 'n_links': 605, 'speed_max': 111}],
                'NYC': [{'step_size': 60, 'n_links': 2, 'speed_max': 267}],
                'Los': [{'step_size': 5, 'n_links': 207, 'speed_max': 70}],
                'BJ':[{'step_size': 30, 'n_links': 2, 'speed_max': 1292}],
                'BJs':[{'step_size': 30, 'n_links': 2, 'speed_max': 1292}],
                'amapHK':[{'step_size': 2, 'n_links': 1, 'speed_max': 1}] }

    basedModels=['LSTM', 'BiLSTM', 'GRU']

    params={'location': args.location,
            'cuda': args.cuda,
            'isBi': args.isBi,
            'mode': args.mode,
            'region':  args.region,
            'batch_size': args.batch_size,
            'model_name': Models[args.flag],
            'total_epoch':args.total_epoch,
            'n_hidden_encoder': args.n_hidden_encoder,
            'n_steps_encoder': args.n_steps_encoder,
            'dropout_rate': args.dropout_rate,
            'learning_rate': args.learning_rate,
            'n_links': pre_data[args.region][0]['n_links'],
            'speed_max': pre_data[args.region][0]['speed_max'],
            'baseModel': basedModels[args.isBi],
            'args': args
            }

    return params

def update_hps(params):

    hps = basic_hyperparams()

    params['n_hidden_encoder'] = params['args'].n_hidden_encoder

    hps.n_links = params['n_links']
    hps.dropout_rate = params['dropout_rate']
    hps.learning_rate = params['learning_rate']
    hps.n_steps_encoder = params['n_steps_encoder']
    hps.n_hidden_encoder = params['n_hidden_encoder']
    hps.n_steps_decoder = params['args'].n_steps_decoder

    return hps, params

def continue_train(params):
    region = params['region']
    location = params['location']
    batch_size = params['batch_size']
    baseModel=params['baseModel']
    model_name = params['model_name']
    total_epoch = params['total_epoch']
    mode = params['mode']
    speed_max = params['speed_max']
    print('batch_size:\t', batch_size)

    # path ='/public/hezhix/DataParse/DurationPre/Data/unScale/%s/speedMatrix/' % location
    if 1:

        np.random.seed(2017)
        # use specific gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = params['cuda']
        os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        session = tf.Session(config = tf_config)
        hps, params = update_hps(params)
        print(params)
        hps.n_output_decoder = hps.n_links
        # if params['isBi'] ==2:
        #     hps.n_hidden_decoder = params['n_hidden_encoder']
        if params['isBi'] == 0:
            hps.n_hidden_decoder = params['n_hidden_encoder']

        if region == 'BJs':
        #     print(params['args'].split_id)
            region = region + '%s' % params['args'].split_i
        #     print(region, '\n')
            # train_data, valid_data, test_data = load_taxiBJ_data_split(batch_size, region, params['args'].split_id)
        # if region == 'BJ':
        #     train_data, valid_data, test_data = load_taxiBJ_data(batch_size, region)
        print(hps)
        path = '/public/hezhix/DataParse/DurationPre/Data/unScale/%s/lstm2lstm_all_amapHK/' % region
        print(path)
        train_data, valid_data, test_data = load_data(path)

        # logdir = './logs/saved_models' \
        # logdir = '/data/zhixhe2/DataParse/lstm2lstm_days/logs' \
        #         +  '/%s-b_%s-nd_%s-np_%s-lr_%4f-dr%4f-hs_%d/' % (region, batch_size, \
        #             1, 1, hps.learning_rate, hps.dropout_rate, hps.n_hidden_encoder)

        logdir = './logs/%s-b_%s-nd_%s-np_%s-lr_%4f-dr%4f-hs_%d/' % (region, batch_size, \
                    1, 1, hps.learning_rate, hps.dropout_rate, hps.n_hidden_encoder)
        print('logdir:\t%s' % logdir)
        model_dir = logdir
        mkdir_file(logdir)
        mkdir_file(model_dir)
        # load speed data
        # speedMatrix = np.load(path + '%s_%s_speedMatrix_without_nan.npy' % (mode, location))
        # print(subpath_region)
        # if region == 'BJ':
        #     train_data, valid_data, test_data = load_taxiBJ_data(batch_size, region)
        # elif region == 'NYC':
        #     train_data, valid_data, test_data = load_bikeNYC_data(batch_size, region)


        num_train = len(train_data[0])
        num_valid = len(valid_data[0])
        num_test = len(test_data[0])
        print('train samples:\t{0}'.format(num_train))
        print('eval samples:\t{0}'.format(num_valid))
        print('test samples:\t{0}'.format(num_test))

        if 1:
            # model construction
            tf.reset_default_graph()
            if params['isBi']==0:
                train_model = LSTM_Model(hps)
            if params['isBi']==1:
                train_model = BiLSTM_Model(hps)
            if params['isBi']==2:
                train_model = LSTM_Model(hps)
            print('build model successfully')

            # # print trainable params
            # for i in tf.trainable_variables():
            #     print(i)

            # print all placeholders
            # hps = [x for x in tf.get_default_graph().get_operations()
            #     if x.type =="Placeholder"]
            # count the parameters in our model
            total_parameters = 0
            for variable in tf.trainable_variables():
                # shape is an array of tf.Dimension
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            print('total parameters: {}'.format(total_parameters))

            # train params
            display_iter = 500
            save_log_iter = 100
            # n_split_valid = 200  # times of splitting validation set
            valid_losses = [np.inf]

            # training process
            print(' ==========================================epcho {}'.format(total_epoch))
            with tf.Session() as sess:
                saver = tf.train.Saver()
                summary_writer = tf.summary.FileWriter(logdir)
                # initialize
                train_model.init(sess)
                
                if os.path.exists(model_dir + 'checkpoint'):        #判断模型是否存在
                    saver.restore(sess, model_dir+'final_model.ckpt')    #存在就从模型中恢复变量                



                iter = 0
                for i in range(total_epoch):
                    print('-------------------epcho {}----------------------'.format(i))
                    train_data = shuffle_data(train_data, hps.n_steps_encoder, hps.n_steps_decoder)
                    print(model_dir)
                    for l in range(0, num_train, batch_size):
                        batch_train_data = [[]] * len(train_data)
                        iter+=1
                        for j in range(len(train_data)): # 10
                            batch_train_data[j] =  train_data[j][l: l + batch_size]
                            train_model.batch_size = batch_train_data[j].shape[0]
                            # print('batch_train_data[j].shape:\t', batch_train_data[j].shape)
                        feed_dict = get_batch_input_dict(train_model, batch_train_data)
                        _, merged_summary = sess.run(
                            [train_model.phs['train_op'], train_model.phs['summary']], feed_dict)

                        if iter % save_log_iter ==0:
                            print(iter / save_log_iter)
                            summary_writer.add_summary(merged_summary, iter)
                        if iter % display_iter ==0:
                        # train_loss = sess.run(train_model.phs['loss'], input_feed_dict)
                        # print('training loss:\t%s' % train_loss)
                            valid_loss = 0
                            batch_valid_data = [[]] * len(valid_data)
                            # print('start epoch-%s valid batch!!!' % i)
                            for l_ in range(0, num_valid, batch_size):
                                batch_valid_data = [[]] * len(valid_data)
                                for j_ in range(len(valid_data)):
                                    batch_valid_data[j_] =  valid_data[j_][l_: l_ + batch_size]
                                    train_model.batch_size = batch_valid_data[j_].shape[0]
                                valid_feed_dict = get_batch_input_dict(train_model, batch_valid_data)

                                batch_loss = sess.run(train_model.phs['loss'], valid_feed_dict)
                                valid_loss +=  batch_loss
                            valid_loss /=  int(num_valid / batch_size)
                            valid_losses.append(valid_loss)
                            valid_loss_sum = tf.Summary(
                                value = [tf.Summary.Value(tag = "valid_loss", simple_value = valid_loss)])
                            summary_writer.add_summary(valid_loss_sum, iter)
                            if valid_loss < min(valid_losses[:-1]):
                                print('iter {}\tvalid_loss = {:.6f}\tmodel saved!!'.format(
                                    iter, valid_loss))
                                saver.save(sess, model_dir +
                                        'model_{}.ckpt'.format(iter))
                                saver.save(sess, model_dir + 'final_model.ckpt')
                            else:
                                print('iter {}\tvalid_loss = {:.6f}\t'.format(
                                    iter, valid_loss))

                print('stop training !!!')

def test_model_all_all(params):
    print('======================++++++++++++++++++test')
    region = params['region']
    location = params['location']
    batch_size = params['batch_size']
    baseModel=params['baseModel']
    model_name = params['model_name']
    total_epoch = params['total_epoch']
    mode = params['mode']
    speed_max = params['speed_max']
    print('batch_size:\t', batch_size)

    if 1:

        np.random.seed(2017)
        # use specific gpu
        # os.environ["CUDA_VISIBLE_DEVICES"] = params['cuda']
        os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        session = tf.Session(config = tf_config)
        hps, params = update_hps(params)
        hps.n_output_decoder = hps.n_links
        # if params['isBi'] ==2:
        #     hps.n_hidden_decoder = params['n_hidden_encoder']
        if region == 'BJs':
        #     print(params['args'].split_id)
            region = region + '%s' % params['args'].split_id
        #     print(region, '\n')
        #     train_data, valid_data, test_data = load_taxiBJ_data_split(batch_size, region, params['args'].split_id)
        path = '/public/hezhix/DataParse/DurationPre/Data/unScale/%s/lstm2lstm_all_amapHK/' % region
        train_data, valid_data, test_data = load_data(path)


        # logdir = './logs/saved_models' \

        # logdir = '/data/zhixhe2/DataParse/lstm2lstm_days/logs' \
        """
        logdir = '/public/hezhix/DataParse/DurationPre/dnn/Baselines/lstm2lstm_amap/scripts/logs' \
                +  '/%s-b_%s-nd_%s-np_%s-lr_%4f-dr%4f-hs_%d/' % (region, batch_size, \
                    1, 1, hps.learning_rate, hps.dropout_rate, hps.n_hidden_encoder)
                    """
        logdir = './logs/%s-b_%s-nd_%s-np_%s-lr_%4f-dr%4f-hs_%d/' % (region, batch_size, \
                    1, 1, hps.learning_rate, hps.dropout_rate, hps.n_hidden_encoder)
        print('logdir:\t%s' % logdir)
        model_dir = logdir
        mkdir_file(logdir)
        mkdir_file(model_dir)
        train_data, valid_data, test_data = load_data(path)

        loop_num = 2
        am_rmses = np.zeros((loop_num))
        am_maes = np.zeros((loop_num))
        am_MAREs = np.zeros((loop_num))
        am_sMAREs = np.zeros((loop_num))
        pm_rmses = np.zeros((loop_num))
        pm_maes = np.zeros((loop_num))
        pm_MAREs = np.zeros((loop_num))
        pm_sMAREs = np.zeros((loop_num))
        decoders = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        thefile = open(model_dir + 'test_res.txt', 'a')
        if os.path.exists(model_dir + 'checkpoint'):        #判断模型是否存在
            num_test = len(test_data[0])
            print('test samples: {0}'.format(num_test))
            # model construction
            tf.reset_default_graph()
            if params['isBi']==0:
                test_model = LSTM_Model(hps)
            if params['isBi']==1:
                test_model = BiLSTM_Model(hps)
            if params['isBi']==2:
                test_model = LSTM_Model(hps)
            saver = tf.train.Saver()
            # # print trainable params
            # for i in tf.trainable_variables():
            #     print(i)
            n_split_test = 200  # times of splitting test set
            # restore model
            print("Starting loading model...")
            with tf.Session() as sess:
                test_model.init(sess)
                saver.restore(sess, model_dir+'final_model.ckpt')    #存在就从模型中恢复变量
                print('restore successfully')

                test_loss = 0
                batch_test_data = [[]] * len(test_data)
                preds = []
                for l_ in range(0, num_test, batch_size):
                    batch_test_data = [[]] * len(test_data)
                    for j_ in range(len(test_data)):
                        batch_test_data[j_] =  test_data[j_][l_: l_ + batch_size]
                        # print(batch_test_data[j_].shape)
                        # print('start batch!!!')
                        # print(batch_test_data[j_].shape)
                    test_feed_dict = get_batch_input_dict(test_model, batch_test_data)
                    batch_preds = sess.run(test_model.phs['preds'], test_feed_dict)
                    batch_preds = np.array(batch_preds)
                    # print(batch_preds.shape) # (168, 256, 2)

                    if preds == []:
                        preds=batch_preds
                        # print(len(preds))
                    else:
                        # print(batch_preds.shape)
                        preds=np.concatenate((preds, batch_preds), axis=1)
                preds = np.transpose(np.array(preds), [1, 0, 2])
                labels = test_data[1]
                # preds = np.transpose(preds, [1, 0, 2])
                labels = labels * speed_max
                preds = preds * speed_max
                # 1896,168,2) (44856,256,2)
                # print(labels.shape)
                # print(preds.shape)
                # preds = np.reshape(preds, [preds.shape[0]*preds.shape[1], -1])*speed_max
                # labels = np.reshape(labels, [labels.shape[0]*labels.shape[1], -1])*speed_max
                print('reshape labels:\t', labels.shape)
                loop_num=2
                pred_path = model_dir + '%s_preds_%s.npz' % (region, model_name)
                gt_path = model_dir + '%s_gt_%s.npz' % (region, model_name)
                print('pred_path', pred_path)
                print('gt_path', gt_path)
                np.savez_compressed(pred_path, pred=preds)
                np.savez_compressed(gt_path, gt=labels)
                test_maes = mean_absolute_error(labels, preds)
                test_rmses = root_mean_squared_error(labels, preds)
                test_MAREs = mean_absolute_relative_error(labels, preds)
                test_sMAREs = sy_mean_absolute_relative_error(labels, preds)*100
                print('test_maes:\t', test_maes)
                print('test_rmses:\t', test_rmses)
                thefile.write('all\t%s\t%s\t' % \
                    (region, model_name))
                thefile.write('epoch-%s\t%.6f\t%.6f\t%.6f\t%.6f'  % \
                    (total_epoch, speed_max, test_rmses, test_maes, test_MAREs))
                thefile.write('\n')
        else:
            print('%s\nchedkpoint not exists!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!' % model_dir)

def test_model_rush(params):

    print('======================++++++++++++++++++test')
    subpath = params['subpath']
    location = params['location']
    batch_size = params['batch_size']
    baseModel=params['baseModel']
    model_name = params['model_name']
    total_epoch = params['total_epoch']
    # link_ids = params['link_ids']
    mode = params['mode']
    subpath_region = params['subpath_region']
    if subpath in ['HongKong_HK', 'HongKong_KL', 'HongKong_HK-KL']:
        params['speed_max'] = 88
    if subpath in ['HongKong_ST', 'HongKong_TM']:
        params['speed_max'] = 111

    speed_max = params['speed_max']

    am_tmp_path = './logs/rush_hour/am_%s_results_%s_%s.txt' % (params['location'], baseModel, params['subpath'])
    pm_tmp_path = './logs/rush_hour/pm_%s_results_%s_%s.txt' % (params['location'], baseModel, params['subpath'])
    mkdir_file(am_tmp_path)
    mkdir_file(pm_tmp_path)
    am_thefile = open(am_tmp_path, 'a')
    pm_thefile = open(pm_tmp_path, 'a')

    # for n_steps_decoder in [3, 6, 9, 12]:
    if 1:

        n_steps_decoder = params['args'].n_steps_decoder
        np.random.seed(2017)
        # use specific gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = params['cuda']
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        session = tf.Session(config = tf_config)

        link_ids = params['link_ids']
        hps, params = update_hps(params)

        if params['isBi'] ==2:
            hps.n_hidden_decoder = params['n_hidden_encoder']
        if params['isBi'] == 0:
            hps.n_hidden_decoder = params['n_hidden_encoder']
        if params['isBi'] == 1:
            hps.n_hidden_decoder = 2*params['n_hidden_encoder']
        hps.n_steps_decoder = n_steps_decoder
        hps.n_output_decoder = hps.n_links
        print(hps)
        if hps.static_attr_flag and hps.grid_flag:
            model_name += '-gr'
        elif hps.static_attr_flag:
            model_name += '-g'
        elif hps.grid_flag:
            model_name += '-r'
        print('model_name:\t%s' % model_name)
        logdir = '/data/zhixhe2/DataParse/GeoMan_my/STNN/logs/{}_721/{}/{}-{}/encoder_{}-decoder_{}/'.format(location,
                                                                                                    subpath,
                                                                                                    subpath_region,
                                                                                                    baseModel,
                                                                                                    hps.n_steps_encoder,
                                                                                                    hps.n_steps_decoder) \
                    + 'epoch_{}-{}/grid_{}_{}_{}-ext_{}-static_{}_{}/{}-{}-{}-{}-{:.3f}-{:.4f}/'.format(total_epoch,
                                                                                                    params['n_split_link'],
                                                                                                    hps.grid_flag,
                                                                                                    hps.n_grid,
                                                                                                    hps.grid_cnn_output_size,
                                                                                                    hps.external_flag,
                                                                                                    hps.static_attr_flag,
                                                                                                    hps.n_static_attr_output_size,
                                                                                                    model_name,
                                                                                                    batch_size,
                                                                                                    hps.n_hidden_decoder,
                                                                                                    hps.n_stacked_layers,
                                                                                                    hps.dropout_rate,
                                                                                                    hps.learning_rate)
        model_dir = logdir

        input_path = '/public/hezhix/DataParse/DurationPre/Data/Scale/%s_721/' % (location)
        input_path1 = input_path + '%s_steps_encoder/%s_steps_decoder/%s/%s/' %  \
                                       (hps.n_steps_encoder, hps.n_steps_decoder, subpath, mode)
        # state_path = input_path + '%s_steps_decoder/%s/%s/' %  (hps.n_steps_decoder, subpath_region, mode)
        state_path = '/public/hezhix/DataParse/DurationPre/Data/Scale/%s_721/%s_steps_encoder/1_steps_decoder/%s/%s/' % (location, hps.n_steps_encoder, subpath, mode)
        # print(model_dir)
        print('-----------------1')

        loop_num = len(link_ids)
        am_rmses = np.zeros((loop_num))
        am_maes = np.zeros((loop_num))
        am_MAREs = np.zeros((loop_num))
        am_sMAREs = np.zeros((loop_num))
        pm_rmses = np.zeros((loop_num))
        pm_maes = np.zeros((loop_num))
        pm_MAREs = np.zeros((loop_num))
        pm_sMAREs = np.zeros((loop_num))

        if os.path.exists(model_dir+'saved_models/checkpoint'):        #判断模型是否存在
            test_data = load_data(input_path, input_path1,
                              'test',
                              hps.n_steps_encoder,
                              hps.n_steps_decoder,
                              hps.n_grid,
                              params['region'])
            global_inputs_test, self_attn_states_test = load_global_inputs(input_path1,
                                                                state_path,
                                                                'test',
                                                                hps.n_steps_encoder,
                                                                hps.n_steps_decoder)
            static_attr = load_link_static_attr_pickle_to_df(params['region'])

            num_test = len(test_data[0])
            print('test samples: {0}'.format(num_test))

            # model construction
            tf.reset_default_graph()
            if params['isBi']==0:
                model = LSTM_Model(hps)
            if params['isBi']==1:
                model = BiLSTM_Model(hps)
            if params['isBi']==2:
                model = LSTM_Model(hps)
            saver = tf.train.Saver()


            n_split_test = 500  # times of splitting test set

            # restore model
            print("Starting loading model...")
            with tf.Session() as sess:
                model.init(sess)
                saver.restore(sess, model_dir+'saved_models/final_model.ckpt')    #存在就从模型中恢复变量
                print('restore successfully')
                test_loss = 0
                test_indexes = np.int64(
                    np.linspace(0, num_test, n_split_test))

                for k in range(n_split_test - 1):
                    feed_dict = get_valid_batch_feed_dict(
                            model, test_indexes, k, test_data, global_inputs_test, self_attn_states_test, static_attr)
                    batch_preds = sess.run(model.phs['preds'], feed_dict)

                    if k==0:
                        preds=batch_preds
                    else:
                        preds=np.concatenate((preds, batch_preds), axis=1)
                preds = np.transpose(preds, [1, 0, 2])
                preds=preds*speed_max # num_test, n_steps_decoder, loop_num
                labels = test_data[4]*speed_max

                # try:
                #     np.save('./logs/%s_%s_preds.npy' % (subpath_region, baseModel), preds)
                #     np.save('./logs/%s_%s_labels.npy' % (subpath_region, baseModel), labels)
                indices  = range(labels.shape[0])


                # prepare indices for am and pm
                am_idx = []
                pm_idx = []
                for i in range(1, int(labels.shape[0]/144)):
                    am_idx.append(list(range(97+42+i*144, 97+42+18+i*144)))
                    pm_idx.append(list(range(97+96+i*144, 97+96 + 18 + i*144)))
                am_idx = np.array(am_idx)
                am_idx = am_idx.flatten()
                pm_idx = np.array(pm_idx)
                pm_idx = pm_idx.flatten()

                am_idx = am_idx.tolist()
                pm_idx = pm_idx.tolist()
                # print('labels.shape')
                # print(labels.shape)

                # print(preds.shape)

                # for loop_idx in range(loop_num):
                #     true_speed = labels[:, :, loop_idx]
                #     preds_speed = preds[:, :, loop_idx]
                #     # ground truth
                #     am_true_speed = np.reshape(true_speed[am_idx], [true_speed[am_idx].shape[0] * n_steps_decoder])
                #     pm_true_speed = np.reshape(true_speed[pm_idx], [true_speed[pm_idx].shape[0] * n_steps_decoder])

                #     # prediction results
                #     am_preds_speed = np.reshape(preds_speed[am_idx], [preds_speed[am_idx].shape[0] * n_steps_decoder])
                #     pm_preds_speed = np.reshape(preds_speed[pm_idx], [preds_speed[pm_idx].shape[0] * n_steps_decoder])
                #     if loop_idx == 0:
                #         print(true_speed[am_idx].shape) # 636* 12
                #         print(true_speed[pm_idx].shape)

                #         print(pm_preds_speed.shape)
                #         print(am_preds_speed.shape)
                for horizon in [2, 5, 8, 11]:
                    for loop_idx in range(loop_num):
                        true_speed = labels[:, horizon, loop_idx]
                        # print(true_speed.shape)
                        preds_speed = preds[:, horizon, loop_idx] # pred of the last step
                        # print(preds_speed.shape)

                        # ground truth
                        am_true_speed = np.reshape(true_speed[am_idx], [true_speed[am_idx].shape[0]])
                        pm_true_speed = np.reshape(true_speed[pm_idx], [true_speed[pm_idx].shape[0]])

                        # prediction results
                        am_preds_speed = np.reshape(preds_speed[am_idx], [preds_speed[am_idx].shape[0]])
                        pm_preds_speed = np.reshape(preds_speed[pm_idx], [preds_speed[pm_idx].shape[0]])
                        if loop_idx == 0:
                            print(true_speed[am_idx].shape) # 636* 12
                            print(true_speed[pm_idx].shape)

                            print(pm_preds_speed.shape)
                            print(am_preds_speed.shape)
                            # print(am_true_speed)
                            # print(pm_true_speed)
                            # print(pm_preds_speed)
                            # print(am_preds_speed)
                        # metrics for am
                        # print(mean_absolute_error(am_true_speed, am_preds_speed))
                        am_maes[loop_idx] = mean_absolute_error(am_true_speed, am_preds_speed)
                        am_rmses[loop_idx] = root_mean_squared_error(am_true_speed, am_preds_speed)
                        am_MAREs[loop_idx] = mean_absolute_relative_error(am_true_speed, am_preds_speed)
                        am_sMAREs[loop_idx] = sy_mean_absolute_relative_error(am_true_speed, am_preds_speed)
                        # print(mean_absolute_error(pm_true_speed, pm_preds_speed))
                        # print('\n')
                        # metrics for pm
                        pm_maes[loop_idx] = mean_absolute_error(pm_true_speed, pm_preds_speed)
                        pm_rmses[loop_idx] = root_mean_squared_error(pm_true_speed, pm_preds_speed)
                        pm_MAREs[loop_idx] = mean_absolute_relative_error(pm_true_speed, pm_preds_speed)
                        pm_sMAREs[loop_idx] = sy_mean_absolute_relative_error(pm_true_speed, pm_preds_speed)

                    print('metric shape')
                    # print(pm_maes)
                    # print(am_maes)

                    pm_thefile.write('\n%s\tgrid_%s_%s_%s\text_%s\tstatic_%s_%s\t\n%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t' % (\
                                    horizon + 1, params['args'].grid_flag, hps.n_grid, hps.grid_cnn_output_size,\
                                    hps.external_flag, params['args'].static_attr_flag, hps.n_static_attr_output_size,\
                                    params['subpath'], params['subpath_region'], params['n_steps_encoder'], \
                                    n_steps_decoder, model_name, baseModel, hps.external_flag, batch_size, \
                                    hps.n_hidden_encoder, hps.dropout_rate, hps.learning_rate, hps.n_stacked_layers))
                    pm_thefile.write('epoch-%s\t%.6f\t%.6f\t%.6f\t%.6f\t%f\n'  % \
                                    (total_epoch, speed_max, np.mean(am_rmses), np.mean(am_maes), \
                                    np.mean(am_MAREs), np.mean(am_sMAREs)))


                    pm_thefile.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t' % \
                                    (params['subpath'], params['subpath_region'], params['n_steps_encoder'],\
                                    n_steps_decoder, model_name, baseModel, hps.external_flag, batch_size, \
                                    hps.n_hidden_encoder, hps.dropout_rate, hps.learning_rate, hps.n_stacked_layers))
                    pm_thefile.write('epoch-%s\t%.6f\t%.6f\t%.6f\t%.6f\t%f\n'  % \
                                    (total_epoch, speed_max, np.mean(pm_rmses), np.mean(pm_maes), \
                                    np.mean(pm_MAREs), np.mean(pm_sMAREs)))


                    am_thefile.write('%s\tgrid_%s_%s_%s\text_%s\tstatic_%s_%s\tMean\tepoch-%s\t%.6f\t%.6f\t%.6f\t%.6f\t%f\n'  % \
                                    (horizon+1, params['args'].grid_flag,  hps.n_grid, hps.grid_cnn_output_size, \
                                    hps.external_flag, params['args'].static_attr_flag, hps.n_static_attr_output_size,\
                                    total_epoch, speed_max, 0.5*(np.mean(am_rmses) + np.mean(pm_rmses)),\
                                    0.5*(np.mean(am_maes)+ np.mean(pm_maes)), 0.5* ( np.mean(am_MAREs)+ np.mean(pm_MAREs)),\
                                    0.5*(np.mean(am_sMAREs)+np.mean(pm_sMAREs))))

                    print('epoch-%s\t%.6f\t%.6f\t%.6f\t%.6f\t%f\n'  % \
                                    (total_epoch, speed_max, np.mean(am_rmses), np.mean(am_maes), \
                                    np.mean(am_MAREs), np.mean(am_sMAREs)))
                    print('epoch-%s\t%.6f\t%.6f\t%.6f\t%.6f\t%f\n'  % \
                                    (total_epoch, speed_max, np.mean(pm_rmses), np.mean(pm_maes), \
                                    np.mean(pm_MAREs), np.mean(pm_sMAREs)))
        else:
            print('final_model not exists..')

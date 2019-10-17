#!/usr/bin/python
# -*- coding: utf-8 -*-

from lstm2lstm_func_model import test_model_all_all, test_model_rush

from lstm2lstm_func_model import mean_absolute_relative_error
from lstm2lstm_func_model import mean_absolute_error
from lstm2lstm_func_model import root_mean_squared_error
from lstm2lstm_func_model import sy_mean_absolute_relative_error

from lstm2lstm_func_model import load_params
from lstm2lstm_func_model import continue_train
from lstm2lstm_func_model import mkdir_file


import argparse
parser = argparse.ArgumentParser()
# given default values
parser.add_argument("--subpath", type=str, default='ALL', help="subpath")
parser.add_argument("--isBi", type=int, default=0, help='LSTM: 0 or BiLSTM: 1')
parser.add_argument("--mode", type=str, default='re_range', help='re_range or all')
parser.add_argument("--n_hidden_encoder", type=int, default=256, help="n_hidden_encoder")
parser.add_argument("--link_ratio", type=float, default= 1.0, help='link_ratio')
parser.add_argument("--n_steps_encoder", type=int, default=300, help="n_steps_encoder")
parser.add_argument("--n_steps_decoder", type=int, default=300, help="n_steps_decoder")
parser.add_argument("--add_epoch", type=int, default=100, help="add_epoch")

# traing and test setting
parser.add_argument("--opt", type=int, help='train or/and test')
parser.add_argument("--cuda", type=str, help='cuda: which gpu')
parser.add_argument("--flag", type=int, help='flag: which model')
parser.add_argument("--total_epoch", type=int, help='total_epoch')
parser.add_argument("--batch_size", type=int, default=256, help="batch_size")
parser.add_argument("--learning_rate", type=float, default=0.001, help='learning_rate')
parser.add_argument("--dropout_rate", type=float, default=0, help='dropout_rate')

# dataset setting
parser.add_argument("--location", type=str, default='amapHK', help="location, HongKong, Seattle, NYC, Los")
parser.add_argument("--region", type=str, default='amapHK', help='start sub-network')
parser.add_argument("--split_id", type=int, default=0, help='split_id')



args = parser.parse_args()

if __name__=='__main__':
    params=load_params(args)

    if args.opt==2:
        print('============================ = train')
        continue_train(params)
        print('============================ = test')
        test_model_all_all(params)
    if args.opt ==  4:
        print('============================ = test')
        test_model_all_all(params)
        # test_model_rush(params)
    if args.opt ==  5:
        # test_model_rush(params)
        test_model_all_mean_links(params)

import os
import io
import six
import csv
import keras
import shutil
import numpy as np
import configparser
import argparse
import tensorflow as tf
from read_data import read_data
from my_regularizer import *
from keras import backend as K
from keras.layers import Dense, Dropout,LeakyReLU
from generate_cfg import generate_cfg
from keras.models import Sequential
from collections import Iterable
from collections import OrderedDict
from sklearn import preprocessing, model_selection
from L0_log import csv_logger



def create_model(cfg):
    model = Sequential()
    num_layer = len(cfg.layer_node)
    model.add(Dense(units=cfg.layer_node[0],
                    input_shape=(cfg.input_shape,),
                    activation="relu",
                    kernel_regularizer=reg_loss(cfg.reg_type, cfg.reg_lambda[0], cfg.sigma)))
    model.add(Dropout(cfg.dropout_rate))
    for i in range(1, num_layer-1):
        model.add(Dense(units=cfg.layer_node[i],
                        activation="relu",
                        kernel_regularizer=reg_loss(cfg.reg_type, cfg.reg_lambda[i], cfg.sigma)))
        model.add(Dropout(cfg.dropout_rate))
    model.add(Dense(units=cfg.layer_node[num_layer-1],
                    activation="softmax",
                    kernel_regularizer=reg_loss(cfg.reg_type, cfg.reg_lambda[num_layer-1], cfg.sigma)))
    optim = keras.optimizers.Adam(
        lr=cfg.base_learning_rate, decay=cfg.learning_decay)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optim, metrics=['accuracy'])
    return model


class config_parsers():
    def __init__(self, config, sec):
        self.dataset_name = config.get(sec, "dataset_name")
        self.input_shape = config.getint(sec, "input_shape")
        self.layer_node = np.array(
            config[sec]["layer_node"].split(",")).astype("int")
        self.reg_type = config.get(sec, "reg_type")
        self.reg_lambda = np.array(
            config[sec]["reg_lambda"].split(",")).astype("float32")
        self.sigma = config.getfloat(sec, "sigma")
        self.train_epoch = config.getint(sec, "train_epoch")
        self.base_learning_rate = config.getfloat(sec, "base_learning_rate")
        self.learning_decay = config.getfloat(sec, "learning_decay")
        self.batch_size = config.getint(sec, "batch_size")
        self.dropout_rate = config.getfloat(sec, "dropout_rate")


def train_loop(config):
    k_fold_acc = []
    k_fold_val_acc = []
    k_fold_nerous_1 = []
    k_fold_nerous_2 = []
    k_fold_nerous_3 = []
    k_fold_nerous_4 = []
    k_fold_sparse_1 = []
    k_fold_sparse_2 = []
    k_fold_sparse_3 = []
    k_fold_sparse_4 = []

    num_splits = 3

    sec = config.sections()[0]
    print("_____________section [%s]_____________ " % sec)
    cfg = config_parsers(config, sec)
    save_path = "log/%s/" % sec
    os.makedirs(save_path)
    X, Y = read_data(cfg)
    x_train_val, x_test, y_train_val, y_test = model_selection.train_test_split(
        X, Y, test_size=0.10, shuffle=True)
    k_fold = model_selection.KFold(n_splits=num_splits, shuffle=False)
    k_flod_index = 1
    for train_index, test_index in k_fold.split(x_train_val):
        k_flod_save_path = save_path+"/%d/" % k_flod_index
        os.mkdir(k_flod_save_path)
        print("<---------  %d fold cross validation , %d step ---------> " %(num_splits,k_flod_index))
        model = create_model(cfg)
        X_train, X_test = x_train_val[train_index], x_train_val[test_index]
        Y_train, Y_test = y_train_val[train_index], y_train_val[test_index]
        lr_reduce = keras.callbacks.ReduceLROnPlateau(
            "val_loss", factor=0.1, patience=2)
        early_stop = keras.callbacks.EarlyStopping("val_loss", patience=5)
        csv_log = csv_logger(cfg, sec, k_flod_index,
                             save_path+"val_10_fold.csv", append=True)
        model.fit(X_train, Y_train,
                  epochs=cfg.train_epoch,
                  batch_size=cfg.batch_size,
                  verbose=2,
                  validation_data=(X_test, Y_test),
                  callbacks=[csv_log, early_stop, lr_reduce])

        score = model.evaluate(x_test, y_test)
        k_fold_acc.append(csv_log.train_end_acc)
        k_fold_val_acc.append(score[1])

        k_fold_nerous_1.append(csv_log.train_end_neuron[0])
        k_fold_nerous_2.append(csv_log.train_end_neuron[1])
        k_fold_nerous_3.append(csv_log.train_end_neuron[2])
        k_fold_nerous_4.append(csv_log.train_end_neuron[3])

        k_fold_sparse_1.append(csv_log.train_end_sparse[0])
        k_fold_sparse_2.append(csv_log.train_end_sparse[1])
        k_fold_sparse_3.append(csv_log.train_end_sparse[2])
        k_fold_sparse_4.append(csv_log.train_end_sparse[3])

        k_flod_index += 1
    acc = sum(k_fold_acc)/num_splits
    val_acc = sum(k_fold_val_acc)/num_splits
    neurons = [sum(k_fold_nerous_1)/num_splits, sum(k_fold_nerous_2)/num_splits,
               sum(k_fold_nerous_3)/num_splits, sum(k_fold_nerous_4)/num_splits]
    spares = [sum(k_fold_sparse_1)/num_splits, sum(k_fold_sparse_2)/num_splits,
              sum(k_fold_sparse_3)/num_splits, sum(k_fold_sparse_4)/num_splits]

    return acc, val_acc, neurons, spares

def get_benchmark(data_set):
    global benchmark_acc,benchmark_val_acc,benchmark_neurons_sum,benchmark_sparse_sum
    global benchmark_sparse,benchmark_neurons
    if data_set=="sdd":
        benchmark_acc = 0.98
        benchmark_val_acc = 0.98
        benchmark_sparse = [0.17, 0.36, 0.36, 0.16]
        benchmark_neurons = [48, 35.5, 24.8, 26.3]
    elif data_set=="mnist":
        benchmark_acc = 0.98
        benchmark_val_acc = 0.97
        benchmark_sparse = [0.60, 0.60, 0.34, 0.08]
        benchmark_neurons = [676.4, 311, 249.9, 93.7]
    elif data_set=="covtype":
        benchmark_acc = 0.83
        benchmark_val_acc = 0.83
        benchmark_sparse = [0.04, 0.10, 0.22, 0.14]
        benchmark_neurons = [54.0, 49.0, 47.3, 18.7]
    benchmark_neurons_sum = sum(benchmark_neurons)
    benchmark_sparse_sum = sum(benchmark_sparse)
    return benchmark_acc,benchmark_val_acc,benchmark_neurons_sum,benchmark_sparse_sum


def get_sigma_list(reg_type):
    if reg_type=="type1":
        train_sigma_list=np.linspace(0.0001,0.0009,5).tolist()+np.linspace(0.001,0.009,5).tolist()+np.linspace(0.01,0.09,5).tolist()+np.linspace(0.1,0.9,5).tolist()
    elif reg_type=="type2":
        train_sigma_list=np.linspace(0.0001,0.0009,5).tolist()+np.linspace(0.001,0.009,5).tolist()+np.linspace(0.01,0.09,5).tolist()+np.linspace(0.1,0.9,5).tolist()
    elif reg_type=="type3":
        train_sigma_list = [1,1/2,1/3,1/4,1/5]
    elif reg_type=="type4":
        train_sigma_list=[1/8,2/8,3/8,4/8,5/8,6/8,7/8,8/8,9/8,10/8,11/8,12/8,13/8,14/8,15/8,16/8]
    return train_sigma_list


if __name__ == "__main__":
    """   argparse   """
    parser = argparse.ArgumentParser(description="for example: python --dataset mnist --type type1 --gpus 0,1")
    parser.add_argument('--dataset',type=str,dest="dataset",help="provide mnist,sdd,covtype")
    parser.add_argument("--type",type=str,dest="type",help="regularization function provide type1 ,type2,type3,type4,more info can read my_regularization.py")
    parser.add_argument('--gpus',type=str,dest="gpus",help="GPU ID that you want to use")

    param=parser.parse_args()

    dataset = param.dataset
    if dataset not in ["mnist","sdd","covtype"]:
        raise ValueError("Invalid dataset name ,please choice one of the [mnist,sdd,covtype].",dataset)
    reg_type=param.type = "type1"
    if reg_type not in ["type1","type2","type3","type4"]:
        raise ValueError("Invalid regularization type ,please choice one of the [type1,type2,type3,type4].",reg_type)
    if param.gpus not in ["0","1","0,1"]:
        raise ValueError("Invalid gpu id ",param.gpus)

    os.environ["CUDA_VISIBLE_DEVICES"] = "%s"%(dataset.gpus)

    train_sigma_list=get_sigma_list(reg_type)

    lambda_value_base=[0.00001, 1]

    benchmark_acc,benchmark_val_acc,benchmark_neurons_sum,benchmark_sparse_sum=get_benchmark(dataset)

    file_handle = io.open("%s_%s_result.csv" %(dataset, reg_type), 'w', newline='')

    result_writer = csv.writer(file_handle)

    result_writer.writerow(["acc", "val_acc", "sigma", "lambda","n1", "n2", "n3", "n4", "sp1", "sp2", "sp3", "sp4"])

    file_handle.flush()

    flag_fine_tuning=False

    for sigma_value in train_sigma_list:
        flag_fine_tuning=False
        lambda_value = lambda_value_base
        while True:
            if flag_fine_tuning is False:
                med_lambda = (lambda_value[0]+lambda_value[1])/2.0
                lambda_array = [med_lambda, med_lambda, med_lambda, med_lambda]
                if abs(lambda_value[1]-lambda_value[0]) > 0.0001:
                    config = generate_cfg(data=dataset, reg_type=reg_type,sigma_value=sigma_value, lambda_scope=lambda_array)
                    acc, val_acc, neurons, spares = train_loop(config)
                    neurons_sum = sum(neurons)
                    if val_acc <= benchmark_val_acc:
                        lambda_value = [lambda_value[0], med_lambda]
                    elif neurons_sum > benchmark_neurons_sum:
                        lambda_value = [med_lambda, lambda_value[1]]
                    else:
                        result_writer.writerow([acc,val_acc, sigma_value, lambda_array[0],neurons[0], neurons[1], neurons[2], neurons[3],spares[0], spares[1], spares[2], spares[3]])
                        file_handle.flush()
                        lambda_value=np.linspace(lambda_value[0],lambda_value[1],20).tolist()[1:-1]
                        flag_fine_tuning=True
                else:
                    break
   
            elif flag_fine_tuning is True:
                if len(lambda_value) >0:
                    lambda_f=lambda_value.pop()
                    lambda_array=[lambda_f,lambda_f,lambda_f,lambda_f]
                    config = generate_cfg(data=dataset, reg_type=reg_type,sigma_value=sigma_value, lambda_scope=lambda_array)
                    acc, val_acc, neurons, spares = train_loop(config)
                    neurons_sum = sum(neurons)
                    if val_acc > benchmark_val_acc and neurons_sum < benchmark_neurons_sum:
                        result_writer.writerow([acc,val_acc, sigma_value, lambda_array[0],neurons[0], neurons[1], neurons[2], neurons[3],spares[0], spares[1], spares[2], spares[3]])
                        file_handle.flush()
                else:
                    break
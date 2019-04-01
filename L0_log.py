
import numpy as np
import keras
from collections import OrderedDict
import six
import io
from collections import Iterable
import csv
import os

def get_nozero_info(model):
    n = []
    sp = []
    all_weight = model.get_weights()
    for ind, weight in enumerate(all_weight):
        if ind % 2 == 0:
            weight = np.rint((weight * 1000.0)) / 1000.0
            nozero_x, nozero_y = np.nonzero(weight)
            row_zero_count = weight.shape[0]
            for i in range(weight.shape[0]):
                nozero_row = np.nonzero(weight[i])
                if len(nozero_row[0]) == 0:
                    row_zero_count -= 1
                # print(len(nozero_row[0]))
            n.append(row_zero_count)
            sp.append(1-len(nozero_x)/weight.size)
    model.set_weights(all_weight)
    sp_print = []
    for sp_tmp in sp:
        sp_print.append(round(sp_tmp, 2))
    print(n, sp_print)
    return n, sp


class CustomDialect(csv.excel):
    delimiter = ","


class csv_logger(keras.callbacks.Callback):

    def __init__(self, cfg, sec, k_fold, filename, separator=',', append=False):
        self.cfg = cfg
        self.sep = separator
        self.filename = filename
        self.append = append
        self.num_layer = len(cfg.layer_node)
        self.writer = None
        self.keys = None
        self.append_header = True
        self.file_flags = ''
        self._open_args = {'newline': '\n'}
        self.section = sec
        self.k_fold_index = k_fold
        self.row_dict = OrderedDict()
        self.train_end_acc = 0.0
        self.train_end_val_acc = 0.0
        self.train_end_sparse = []
        self.train_end_neuron = []
        super(csv_logger, self).__init__()

    def on_train_begin(self, logs=None):
        if self.append:
            if os.path.exists(self.filename):
                with open(self.filename, 'r' + self.file_flags) as f:
                    self.append_header = not bool(len(f.readline()))
            mode = 'a'
        else:
            mode = 'w'
        self.csv_file = io.open(self.filename,
                                mode + self.file_flags,
                                **self._open_args)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, six.string_types):
                return k
            elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k

        if self.keys is None:
            self.keys = sorted(logs.keys())
            if "lr" in self.keys:
                self.keys.remove("lr")

        if self.model.stop_training:
            # We set NA so that csv parsers do not fail for this last epoch.
            logs = dict([(k, logs[k] if k in logs else 'NA')
                         for k in self.keys])

        if not self.writer:
            class CustomDialect(csv.excel):
                delimiter = self.sep
            fieldnames = ["section"]+["fold_index"] + \
                ['epoch'] + ["type"]+["sigma"] + self.keys
            for i in range(self.num_layer):
                fieldnames += ["reg_%s" % (str(i+1))]
                fieldnames += ["n%s" % (str(i+1))]
                fieldnames += ["sp%s" % (str(i+1))]
            self.writer = csv.DictWriter(
                self.csv_file, fieldnames=fieldnames, dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()

        self.row_dict["section"] = self.section
        self.row_dict['epoch'] = epoch
        self.row_dict["type"] = self.cfg.reg_type
        self.row_dict["sigma"] = self.cfg.sigma
        self.row_dict["fold_index"] = self.k_fold_index

        n, sp = get_nozero_info(self.model)
        for i in range(self.num_layer):
            self.row_dict["reg_%s" % (str(i+1))] = self.cfg.reg_lambda[i]
            self.row_dict["n%s" % (str(i+1))] = n[i]
            self.row_dict["sp%s" % (str(i+1))] = sp[i]

        self.row_dict.update(
            (key, handle_value(logs[key])) for key in self.keys)

    def on_train_end(self, logs=None):
        # all_weight = self.model.get_weights()
        model_save_path = "log/%s/%d" % (self.section, self.k_fold_index)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        self.model.save_weights(model_save_path+"/weight.h5")
        # for ind, weight in enumerate(all_weight):
        #     if ind % 2 == 0:
        #         weight = np.rint((weight * 1000.0)) / 1000.0
        #         np.savetxt(weight_save_path+"/layer_%d.txt" % ((ind/2)+1), weight,
        #                    header="shape [%d,%d]" % (weight.shape[0], weight.shape[1]))
        self.writer.writerow(self.row_dict)
        self.csv_file.flush()
        self.csv_file.close()
        n, sp = get_nozero_info(self.model)
        self.train_end_neuron = n
        self.train_end_sparse = sp
        self.train_end_acc = self.row_dict["acc"]
        self.train_end_val_acc = self.row_dict["val_acc"]
        self.writer = None

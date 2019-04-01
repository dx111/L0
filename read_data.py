import gzip
import io
import os
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection
from tensorflow.python.platform import gfile


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(f):
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in MNIST image file: %s' % (magic, f.name))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def extract_labels(f):
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                'Invalid magic number %d in MNIST label file: %s' % (magic, f.name))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return labels


def load_mnist():
    TRAIN_IMAGES = 'data/MNIST/train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'data/MNIST/train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 'data/MNIST/t10k-images-idx3-ubyte.gz'
    TEST_LABELS = 'data/MNIST/t10k-labels-idx1-ubyte.gz'
    with gfile.Open(TRAIN_IMAGES, 'rb') as f:
        train_images = extract_images(f)
    with gfile.Open(TRAIN_LABELS, 'rb') as f:
        train_labels = extract_labels(f)
    with gfile.Open(TEST_IMAGES, 'rb') as f:
        test_images = extract_images(f)
    with gfile.Open(TEST_LABELS, 'rb') as f:
        test_labels = extract_labels(f)
    data = np.concatenate([train_images, test_images]).reshape(70000, -1)
    label = np.concatenate([train_labels, test_labels]).reshape(-1, 1)
    dataset = np.concatenate([data, label], axis=1)
    return dataset


def load_iris():
    df=pd.read_csv("data/iris.data")
    df.replace("Iris-setosa",0)
    df.replace("Iris-versicolor",1)
    df.replace("Iris-virginica",2)
    dataset=np.array(df)
    return dataset

def load_sonal():
    df=pd.read_csv("data/sonar.all-data")
    df.replace("R",0)
    df.replace("M",1)
    dataset=np.array(df)
    return dataset

def load_vertebral():
    df=pd.read_csv("data/vertebral_column_data/column_3C.dat",sep=" ")
    df.replace("DH",0)
    df.replace("SL",1)
    df.replace("NO",2)
    dataset=np.array(df)
    return dataset

def load_vehicle():
    df=pd.DataFrame()
    for i in range(ord("a"),ord("i")+1):
        tmp=pd.read_csv("data/vehicle/xa%s.dat"%chr(i),sep=" ",header=None)
        df=df.append(tmp,ignore_index=True)
    df.replace("bus",0)
    df.replace("saab",1)
    df.replace("van",2)
    df.replace("opel",3)
    dataset=np.array(df)
    return dataset

def load_wbc():
    data=np.loadtxt("data/wbc/breast-cancer-wisconsin.data",delimiter=",")
    print(data)
    dataset=data[:,1:]
    print(dataset)
    return dataset

def read_data(cfg):
    global num_feature, dataset
    if cfg.dataset_name == "sdd":
        dataset = np.loadtxt("data/Sensorless_drive_diagnosis.csv", delimiter=",")
    elif cfg.dataset_name == "mnist":
        dataset = load_mnist()
    elif cfg.dataset_name == "covtype":
        dataset = np.loadtxt("data/covtype.data", delimiter=",")
    elif cfg.dataset_name=="iris":
        dataset=load_iris()
    elif cfg.dataset_name=="liver":
        dataset=np.loadtxt("data/liver.csv",delimiter=",")
    elif cfg.dataset_name=="glass":
        data=np.loadtxt("data/glass.data",delimiter=",")
        dataset=data[:,1:]
    elif cfg.dataset_name=="wine":
        dataset=np.loadtxt("data/wine.data",delimiter=",")
        dataset[:,[13,0]]=dataset[:,[0,13]]
    elif cfg.dataset_name=="sonal":
        dataset=load_sonal()
    elif cfg.dataset_name=="spect":
        dataset1=np.loadtxt("data/spect/SPECT.train",delimiter=",")
        dataset2=np.loadtxt("data/spect/SPECT.test",delimiter=",")
        dataset=np.concatenate([dataset1,dataset2])
        dataset[:,[0,22]]=dataset[:,[22,0]]
    elif cfg.dataset_name=="vertebral":
        dataset=load_vertebral()
    elif cfg.dataset_name=="vehicle":
        dataset=load_vehicle()
    elif cfg.dataset_name=="wbc":
        dataset=load_wbc()


    num_feature=cfg.input_shape
    X = dataset[:, 0:num_feature]
    y = dataset[:, num_feature]
    X = X.astype("float32")
    # X/=255
    X = preprocessing.scale(X)
    #X = preprocessing.MinMaxScaler(feature_range=(-1, +1)).fit_transform(X)
    # one hot encoder
    label_encoder = preprocessing.LabelEncoder()
    label = label_encoder.fit_transform(y.ravel())
    label = np.array([label]).T
    one_hot_encoder = preprocessing.OneHotEncoder()
    y = one_hot_encoder.fit_transform(label).toarray()

    return X, y
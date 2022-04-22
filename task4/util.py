import csv
import numpy as np
import math


def load_csv(path):
    csv_reader = csv.reader(open(path, mode='r', encoding='UTF-8'))
    train_data = []
    train_data_label = []
    test_data = []
    test_data_label = []
    for idx, line in enumerate(csv_reader):
        data_item = [float(i) for i in line[0:-1]]
        label = [1, 0] if line[-1] == 'g' else [0, 1]
        if idx <= 199:
            train_data.append(data_item)
            train_data_label.append(label)
        else:
            test_data.append(data_item)
            test_data_label.append(label)
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    train_data_label = np.array(train_data_label)
    test_data_label = np.array(test_data_label)
    return train_data, train_data_label, test_data, test_data_label


def standardization(data):
    miu = np.mean(data, axis=1)
    sigma = np.std(data, axis=1)
    miu.shape = (miu.shape[0], 1)
    sigma.shape = (sigma.shape[0], 1)
    return (data - miu) / sigma


def test_onehot_and_label(onehot, label):
    acc = 0
    err = 0
    for idx in range(len(label)):
        if onehot[idx, label[idx]] == 1:
            acc += 1
        else:
            err += 1
    return acc, err


def probit(x):
    def __probit(x):
        return 0.5 * (1 + math.erf(x) / sqrt_2)
    sqrt_2 = 1.4142135623730951
    if type(x) == int or type(x) == float:
        return __probit(x)
    elif type(x) == np.ndarray:
        rev = np.copy(x)
        with np.nditer(rev, op_flags=['readwrite']) as it:
            for v in it:
                v[...] = __probit(int(v))
        return rev

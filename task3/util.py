import numpy as np
import matplotlib.pyplot as plt

def read_data(path):
    file = open(path, mode='r', encoding='utf8')
    ans = []
    lines = file.readlines()
    for line in lines:
        line = line.strip('\n')
        line_split = line.split(',')
        ans.append(list(map(float, line_split.copy())))
    return np.array(ans)

def normalize(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

def squre_loss(y_predict, y_true):
    """
    Squre loss function,
    :y_predict is the predict output value, the shape must be n * m
    :y_true is the excepted output value, the shape must be n * m
    """
    _ = 1
    __ = 1
    if len(y_predict.shape) == 1 or len(y_true.shape) == 1:
        __ = y_predict.shape
        ____ = y_true.shape
    else:
        _, __ = y_predict.shape
        ___, ____ = y_true.shape
        if __ != ____:
            raise ValueError('Error shape of input!')
        elif _ != ___:
            raise ValueError('The input size not same!')
    y_predict = np.array(y_predict)
    y_true = np.array(y_true)

    return np.sum(np.sum(((y_true - y_predict) ** 2), axis=0) / (2 * _)) / __


def fun_distribution(x):
    return -np.sin(x / 5) + np.cos(x)


def test_data_generator(data_len, range, data_fun, miu, sigma):
    x = np.random.uniform(range[0], range[1], data_len)
    y = data_fun(x)
    episilon = np.random.normal(miu, sigma, data_len)
    return x, y + episilon


def plot(x, y, data_fun, range):
    plt.scatter(x, y, color='red')
    true_x = np.linspace(range[0], range[1], 1000)
    true_y = data_fun(true_x)
    plt.plot(true_x, true_y)
    plt.show()


def generate_multi_ploy(x, degree):
    ans = []
    for i in range(1, degree + 1):
        ans.append(np.power(x, i))
    return np.array(ans).reshape(degree, x.shape[1])

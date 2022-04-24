import math
import numpy as np
from util import probit, cross_entropy, softmax


class LinearClassifier:
    def __init__(self, input_size, class_size):
        self.__input_size = input_size
        self.__class_size = class_size
        self.__weight = np.random.rand(class_size, input_size + 1)
        self.__default_activation = 'none'

    def __pre_set_input_data(self, input_data):
        return np.insert(input_data, 0, 1, axis=1)

    def least_square_learn(self, input_data, labels):
        self.__default_activation = 'none'
        input_data = self.__pre_set_input_data(input_data)
        self.__weight = np.dot(np.dot(np.linalg.pinv(np.dot(input_data.T, input_data)), input_data.T), labels).T

    def probit_learn(self, input_data, labels, lr=0.001, max_epoch=200, batch_size=16, logging_epoch_interval=200, show_log=False):
        self.__default_activation = 'probit'
        input_data = self.__pre_set_input_data(input_data)
        input_size, feat_size = input_data.shape
        batch_num = input_size // batch_size
        batch_num = batch_num + 1 if batch_num * batch_size < input_size else batch_num
        for epoch in range(max_epoch):
            loss_sum = 0
            for batch_iter in range(batch_num):
                start_idx = batch_iter * batch_size
                end_idx = min(start_idx + batch_size, input_size)
                batch_data = input_data[start_idx:end_idx]
                batch_label = labels[start_idx:end_idx]
                curr_batch_size = end_idx - start_idx
                batch_output = probit(np.dot(batch_data, self.__weight.T))
                loss_sum += cross_entropy(batch_output, batch_label)
                grad_w = np.dot((batch_output - batch_label).T, batch_data) / curr_batch_size
                self.__weight -= lr * grad_w
            if epoch % logging_epoch_interval and show_log:
                print(f'Epoch: {epoch} / {max_epoch}, loss is {loss_sum / batch_num}')


    def proba_generative_learn(self, input_data, labels):
        self.__default_activation = 'softmax'
        input_data_by_label = {}
        for idx in range(len(labels)):
            label = labels[idx]
            if label not in input_data_by_label:
                input_data_by_label[label] = []
            input_data_by_label[label].append(input_data[idx])
        for label in input_data_by_label:
            input_data_by_label[label] = np.array(input_data_by_label[label])
        miu_by_labels = {}
        total_sigma = np.zeros((self.__input_size, self.__input_size))
        total_input_size = input_data.shape[0]
        class_proba = []
        for label in input_data_by_label:
            data = input_data_by_label[label]
            miu = np.mean(data, axis=0, keepdims=True)
            miu_by_labels[label] = miu
            sigma = np.cov(data.T)
            label_size = data.shape[0]
            class_proba.append(label_size / total_input_size)
            total_sigma = class_proba[-1] * sigma + total_sigma
        sigma_inv = np.linalg.pinv(total_sigma)
        for label in miu_by_labels:
            miu = miu_by_labels[label]
            self.__weight[label, 1:] = np.dot(sigma_inv, miu.T).squeeze()
            self.__weight[label, 0] = -0.5 * np.dot(np.dot(miu, sigma_inv), miu.T) + math.log(class_proba[label])


    def predict(self, input_data, return_labels=False, activation=None):
        input_data = self.__pre_set_input_data(input_data)
        if activation is None:
            activation = self.__default_activation
        if activation == 'none':
            proba = np.dot(input_data, self.__weight.T)
        elif activation == 'probit':
            proba = probit(np.dot(input_data, self.__weight.T))
        elif activation == 'softmax':
            proba = softmax(np.dot(input_data, self.__weight.T))
        if return_labels:
            y = np.argmax(proba, axis=1)
            return y
        else:
            return proba

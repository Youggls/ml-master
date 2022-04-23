import math
import numpy as np
from util import probit, cross_entropy


class LinearClassifier:
    def __init__(self, input_size, class_size):
        self.__input_size = input_size
        self.__class_size = class_size
        self.__weight = np.random.rand(class_size, input_size + 1)

    def __pre_set_input_data(self, input_data):
        return np.insert(input_data, 0, 1, axis=1)

    def least_square(self, input_data, labels):
        input_data = self.__pre_set_input_data(input_data)
        self.__weight = np.dot(np.dot(np.linalg.inv(np.dot(input_data.T, input_data)), input_data.T), labels).T

    def probit_learn(self, input_data, labels, lr=0.001, max_epoch=200, batch_size=16, logging_epoch_interval=200):
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
            if epoch % logging_epoch_interval == 0:
                print(f'Epoch: {epoch} / {max_epoch}, loss is {loss_sum / batch_num}')


    def probit_predict(self, input_data, return_labels=False):
        input_data = self.__pre_set_input_data(input_data)
        proba = probit(np.dot(input_data, self.__weight.T))
        if return_labels:
            y = np.argmax(proba, axis=1)
            return y
        else:
            return proba


    def predict(self, input_data, return_labels=False):
        input_data = self.__pre_set_input_data(input_data)
        proba = np.dot(input_data, self.__weight.T)
        if return_labels:
            y = np.argmax(proba, axis=1)
            return y
        else:
            return proba

import numpy as np


class LinearClassifier:
    def __init__(self, input_size, class_size):
        self.__input_size = input_size
        self.__class_size = class_size
        self.__weight = np.random.rand(class_size, input_size + 1)

    def least_square(self, input_data, labels):
        input_data = np.insert(input_data, 0, 1, axis=1)
        self.__weight = np.dot(np.dot(np.linalg.inv(np.dot(input_data.T, input_data)), input_data.T), labels)

    def predict(self, input_data, return_lables=False):
        input_data = np.insert(input_data, 0, 1, axis=1)
        proba = np.dot(self.__weight.T, input_data.T).T
        if return_lables:
            y = np.argmax(proba, axis=1)
            return y
        else:
            return proba

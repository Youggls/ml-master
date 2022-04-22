import numpy as np
import random

from util import plot, fun_distribution, test_data_generator, squre_loss, generate_multi_ploy
from linear import LinearRegression


if __name__ == '__main__':
    range = [-5, 5]
    data_len = 10
    miu = 0
    sigma = 0.2
    seed = 43
    degree = 3

    np.random.seed(seed)
    random.seed(seed)
    data_x, data_y = test_data_generator(data_len, range, fun_distribution, miu, sigma)
    data_x = data_x.reshape(data_len, 1).T
    data_y = data_y.reshape(data_len, 1).T

    lr_sgd_model = LinearRegression(1, 1)    
    lr_sgd_model.sgd(data_x, data_y, max_turns=1000, lr=0.01)

    lr_normal_equation_model = LinearRegression(1, 1)
    lr_normal_equation_model.normal_equation(data_x, data_y)

    sgd_loss = squre_loss(lr_sgd_model.predict(data_x), data_y)
    normal_equation_loss = squre_loss(lr_normal_equation_model.predict(data_x), data_y)
    print('sgd loss: {}, normal equation loss: {}'.format(sgd_loss, normal_equation_loss))
    if normal_equation_loss > sgd_loss:
        print('sgd is better!')
    else:
        print('normal equation is better!')

    new_x = generate_multi_ploy(data_x, degree)
    multi_poly_lr_model = LinearRegression(degree, 1)
    multi_poly_lr_model.normal_equation(new_x, data_y)
    multi_poly_loss = squre_loss(multi_poly_lr_model.predict(new_x), data_y)
    print('multi_poly_loss: {}'.format(multi_poly_loss))

    plot(data_x, data_y, fun_distribution, range)

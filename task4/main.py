from util import load_csv, onehot2label, standardization, test_onehot_and_label, test_label_and_label
from linear_classifier import LinearClassifier


if __name__ == '__main__':
    
    train_data, train_data_label, test_data, test_data_label = load_csv('./datasets/Ionosphere+Dataset.csv')
    # train_data = standardization(train_data)
    # test_data = standardization(test_data)

    model = LinearClassifier(34, 2)
    model.least_square_learn(train_data, train_data_label)
    res = model.predict(test_data, return_labels=True)
    acc, err = test_onehot_and_label(test_data_label, res)
    print(f'Least square test set accuracy: {acc / (acc + err)}, error rate: {err / (acc + err)}')

    res = model.predict(train_data, return_labels=True)
    acc, err = test_onehot_and_label(train_data_label, res)
    print(f'Least square train set accuracy: {acc / (acc + err)}, error rate: {err / (acc + err)}')


    model = LinearClassifier(34, 2)
    model.probit_learn(train_data, train_data_label, lr=0.0001, max_epoch=20000, batch_size=16, logging_epoch_interval=200, show_log=False)
    res = model.predict(test_data, return_labels=True, activation='probit')
    acc, err = test_onehot_and_label(test_data_label, res)
    print(f'probit test set accuracy: {acc / (acc + err)}, error rate: {err / (acc + err)}')

    res = model.predict(train_data, return_labels=True, activation='probit')
    acc, err = test_onehot_and_label(train_data_label, res)
    print(f'probit train set accuracy: {acc / (acc + err)}, error rate: {err / (acc + err)}')

    model = LinearClassifier(34, 2)
    temp_train_data_label = onehot2label(train_data_label)
    model.proba_generative_learn(train_data, temp_train_data_label)
    res = model.predict(test_data, return_labels=True, activation='softmax')
    acc, err = test_onehot_and_label(test_data_label, res)
    print(f'proba generative test set accuracy: {acc / (acc + err)}, error rate: {err / (acc + err)}')
    res = model.predict(train_data, return_labels=True, activation='softmax')
    acc, err = test_onehot_and_label(train_data_label, res)
    print(f'proba generative train set accuracy: {acc / (acc + err)}, error rate: {err / (acc + err)}')

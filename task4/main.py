from util import load_csv, standardization, test_onehot_and_label
from linear_classifier import LinearClassifier


if __name__ == '__main__':
    model = LinearClassifier(34, 2)
    train_data, train_data_label, test_data, test_data_label = load_csv('./datasets/Ionosphere+Dataset.csv')
    train_data = standardization(train_data)
    test_data = standardization(test_data)
    model.probit_learn(train_data, train_data_label, max_epoch=20000, lr=0.0001)
    res = model.probit_predict(test_data, return_labels=True)
    print(res)
    # acc, err = test_onehot_and_label(test_data_label, res)
    # print(f'Acc: {acc}, Err: {err}')

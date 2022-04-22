from util import load_csv, standardization, test_onehot_and_label
from linear_classifier import LinearClassifier


if __name__ == '__main__':
    model = LinearClassifier(34, 2)
    train_data, train_data_label, test_data, test_data_label = load_csv('./datasets/Ionosphere+Dataset.csv')
    train_data = standardization(train_data)
    test_data = standardization(test_data)
    model.least_square(train_data, train_data_label)
    res = model.predict(test_data, return_lables=True)
    acc, err = test_onehot_and_label(test_data_label, res)
    print(f'Acc: {acc}, Err: {err}')

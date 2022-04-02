import csv
from conceptlearning import ConceptLearning

if __name__ == '__main__':
    csv_file = csv.reader(open('./datasets/data2.csv', mode='r', encoding='UTF-8'))
    attribute_list = []
    value_list = []
    for idx, line in enumerate(csv_file):
        if idx == 0:
            attribute_list = line[0:-1]
        else:
            value_list.append(line)

    model = ConceptLearning(attribute_list)
    for value in value_list:
        if value[-1] == 'Yes':
            model.add_sample(value[0:-1], True)
        else:
            model.add_sample(value[0:-1], False)

    model.find_s_learn()
    model.candidate_elimination()

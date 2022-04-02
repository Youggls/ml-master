import copy
from tabulate import tabulate

class ConceptLearning:
    none = 'Φ'
    any = '?'
    def __init__(self, attributes=[]):
        self.__attribute_list = attributes
        self.__attribute_possible_value_dict = {}
        self.__sample_list = []
        self.__find_s_hyp = [ConceptLearning.none for _ in range(len(self.__attribute_list))]
        self.__candidate_s = [[ConceptLearning.none for _ in range(len(self.__attribute_list))]]
        self.__candidate_g = [[ConceptLearning.any for _ in range(len(self.__attribute_list))]]

        for attribute in self.__attribute_list:
            self.__attribute_possible_value_dict[attribute] = []

    def add_sample(self, value, boolean):
        if len(value) != len(self.__attribute_list):
            raise ValueError("The length of the list is not equal to the number of attributes")
        self.__sample_list.append((value, boolean))
        for idx, sub_value in enumerate(value):
            if sub_value not in self.__attribute_possible_value_dict[self.__attribute_list[idx]]:
                self.__attribute_possible_value_dict[self.__attribute_list[idx]].append(sub_value)

    def __is_consistent(hyp, sample):
        if sample[1] == True:
            for i in range(len(hyp)):
                if hyp[i] != ConceptLearning.any and hyp[i] != sample[0][i]:
                    return False
            return True
        else:
            for i in range(len(hyp)):
                if hyp[i] != ConceptLearning.any and hyp[i] != sample[0][i]:
                    return True
            return False

    def __is_normal(hyp1, hyp2):
        """If hyp1 is more normal to hyp2, return True, if hyp2 is more normal than hyp1 return False, else return None

        Args:
            hyp1 (_type_): _description_
            hyp2 (_type_): _description_

        Returns:
            _type_: _description_
        """
        hyp1_is_normal = False
        hyp2_is_normal = False
        for i in range(len(hyp1)):
            if hyp1[i] != hyp2[i]:
                if hyp1[i] == ConceptLearning.any:
                    hyp1_is_normal = True
                    if hyp2_is_normal:
                        return None
                    continue
                elif hyp1[i] != ConceptLearning.any and hyp2[i] == ConceptLearning.any:
                    if hyp1_is_normal:
                        return None
                else:
                    return None
        if hyp1_is_normal and not hyp2_is_normal:
            return True
        elif not hyp1_is_normal and hyp2_is_normal:
            return False
        else:
            return None

    def __remove_not_consistent(hyp_list, sample):
        remove_hyp_list = []
        for hyp in hyp_list:
            if not ConceptLearning.__is_consistent(hyp, sample):
                remove_hyp_list.append(hyp)
        for hyp in remove_hyp_list:
            hyp_list.remove(hyp)
        return remove_hyp_list

    def find_s_learn(self):
        for i in range(len(self.__sample_list)):
            if self.__sample_list[i][1] == True:
                for j in range(len(self.__sample_list[i][0])):
                    if self.__find_s_hyp[j] == ConceptLearning.none:
                        self.__find_s_hyp[j] = self.__sample_list[i][0][j]
                    elif self.__find_s_hyp[j] != self.__sample_list[i][0][j]:
                        self.__find_s_hyp[j] = ConceptLearning.any
        print('****** FindS Algorithm Result ******')
        print(tabulate([tuple(self.__find_s_hyp)], headers=self.__attribute_list))

    def candidate_elimination(self):
        for sample in self.__sample_list:
            if sample[1] == True:
                ConceptLearning.__remove_not_consistent(self.__candidate_g, sample)
                removed_s_list = ConceptLearning.__remove_not_consistent(self.__candidate_s, sample)
                normal_s_list = []
                for removed_s in removed_s_list:
                    temp_s_hyp = []
                    for (value_s_hyp, value_sample) in zip(removed_s, sample[0]):
                        if value_s_hyp == ConceptLearning.none or value_s_hyp == value_sample:
                            temp_s_hyp.append(value_sample)
                        else:
                            temp_s_hyp.append(ConceptLearning.any)
                    for g_hyp in self.__candidate_g:
                        cmp = ConceptLearning.__is_normal(g_hyp, temp_s_hyp)
                        if cmp is not None and cmp == True:
                            normal_s_list.append(temp_s_hyp)
                            break
                for normal_s in normal_s_list:
                    self.__candidate_s.append(normal_s)
            else:
                ConceptLearning.__remove_not_consistent(self.__candidate_s, sample)
                removed_g_list = ConceptLearning.__remove_not_consistent(self.__candidate_g, sample)
                normal_g_list = []
                for removed_g in removed_g_list:
                    temp_g_hyp_list = []
                    special_idx_list = []
                    for idx, (value_g_hyp, value_sample) in enumerate(zip(removed_g, sample[0])):
                        if value_g_hyp == ConceptLearning.any and value_sample != ConceptLearning.any:
                            special_idx_list.append(idx)
                    for special_idx in special_idx_list:
                        for possible_value in self.__attribute_possible_value_dict[self.__attribute_list[special_idx]]:
                            if possible_value != sample[0][special_idx]:
                                temp_g_hyp = copy.deepcopy(removed_g)
                                temp_g_hyp[special_idx] = possible_value
                                temp_g_hyp_list.append(temp_g_hyp)
                    for temp_g_hyp in temp_g_hyp_list:
                        for s_hyp in self.__candidate_s:
                            cmp = ConceptLearning.__is_normal(temp_g_hyp, s_hyp)
                            if cmp is not None and cmp == True:
                                normal_g_list.append(temp_g_hyp)
                                break
                for normal_g in normal_g_list:
                    self.__candidate_g.append(normal_g)
        print('****** Candidate Elimination Algorithm Result ******')
        print('Candidate G:')
        print(tabulate([tuple(t) for t in self.__candidate_g], headers=self.__attribute_list))
        print('Candidate S:')
        print(tabulate([tuple(t) for t in self.__candidate_s], headers=self.__attribute_list))

from help import argmax
import heapq
import help
from math import log2
import random
import operator


"""
KNN model class.

"""
class KNN_Model:
    """
    knn model class.
    it does not really trains the  data.
    it predict by moving over all train data and finds the k most close to the test example.

    """
    def __init__(self, k):
        self.k = k

    def train(self, train_data):
        self.train_data = train_data

    def __get_neighbors(self, test_example):
        """
        return the k nearest neigbors of test_example in train_data
        """
        distances = []
        for i in range(len(self.train_data)):
            heapq.heappush(distances, (help.hamming_distance(test_example, self.train_data[i][:-1]), i))

        neigbors_indexes = [heapq.heappop(distances)[1] for i in range(self.k)]
        neigbors = [self.train_data[idx] for idx in neigbors_indexes]
        return neigbors


    def predict(self, test_example):
        """
        predict test_example decision by train_data and number of neigbors k.
        """
        # take k nearest neigbors, and calculate their decisions (yes/no).
        k_neigbors = self.__get_neighbors(test_example)
        k_neigbors_decisions = [neigbor[-1] for neigbor in k_neigbors]
        # return the majority decision to be the answer to the test example.
        prediction = max(set(k_neigbors_decisions), key=k_neigbors_decisions.count)

        return prediction


class Naive_Bayes_Model:
    """
    class for naive bayes model classifier.
    its train function analyse probabilites in the train set.
    its predict function use probabilites to calculate the probabilty for each class given the test example.
    """
    def __init__(self, classes, features_by_count):
        self.classes = classes
        self.features_counts = features_by_count


    def train(self, train_data):
        # analyse frequences. gp is list of dicts of lists, so that
        # Pr(xi | Ck)

        # seperate data by class:
        data_by_class = [[] for i in range(len(self.classes))]
        for xy in train_data:
            x = xy[:-1]
            y = xy[-1]
            data_by_class[y].append(x)

        class_probs = [len(data_cls) / len(train_data) for data_cls in data_by_class]

        # for each class, keeps counter for each feature value - then calc probability.

        features_values_probs = []
        for cls in range(len(self.classes)):
            # fvc[cls][feature][feature_value] = count how many feature=feature_value in class=cls
            counts = [[0] * feature_count for feature_count in self.features_counts]
            data_cls = data_by_class[cls]
            for x in data_cls:
                for feature in range(len(x)):
                    counts[feature][x[feature]] += 1

            # from counts to probabilites, i.e. fvp[cls][feature][feature_value] = Pr(xi | Ck)
            probs = [[0] * feature_count for feature_count in self.features_counts]
            for feature in range(len(self.features_counts)):
                for feature_val in range(self.features_counts[feature]):
                    probs[feature][feature_val] = counts[feature][feature_val] / len(data_cls)

            features_values_probs.append(probs)

        self.class_probs = class_probs
        self.features_values_probs = features_values_probs



    def predict(self, test_example):
        """
        calculate Pr(C_yes | text_example), Pr(C_no | test_example) and return the argmax.
        """
        class_probabilities = []

        # Pr(Ck | x) = Pr(Ck) * mult(pr(xi | Ck), i = 0 to i = n)
        for cls in range(len(self.classes)):
            pr = self.class_probs[cls]
            for feature in range(len(self.features_counts)):
                val = test_example[feature]
                pr *= self.features_values_probs[cls][feature][val]

            class_probabilities.append(pr)

        # take class with most probability:
        most_likly_class = argmax(class_probabilities)
        return most_likly_class


class Node:
    """
    """
     
    def __init__(self, feature, leaf=False):
        """
        create new node, as tree or not by demand.
        """

        self.__feature_num, self.__feature_desc, self.__feature_values_desc = feature
        self.__leaf = leaf
        if not leaf:
            self.__branches = {}

    def add_branch(self, feature_val, node):
        self.__branches[feature_val] = node

    
    def is_leaf(self):
        return self.__leaf

    def get_branch(self, feature_val):
        if self.__leaf:
            return None
        else:
            return self.__branches[feature_val]

    def val(self):
        return self.__feature_num


    s = ""
    def __str_recursive(self, indent):
        """
        recursively print the tree.
        """
        if self.__leaf:
            Node.s += ':' + self.__feature_desc
        else:
            for val, node in self.__branches.items():
                Node.s += '\n' + '\t' * indent + '|' * int(indent > 0)
                Node.s += self.__feature_desc + '=' + self.__feature_values_desc[val]
                node.__str_recursive(indent + 1)
                
    
    def str(self):
        Node.s = ""
        self.__str_recursive(0)
        return self.s[:]


class Decision_Tree_Model:
    def __init__(self, classes_names, features_values_names):
        self.classes_names = classes_names
        self.classes = len(classes_names)

        self.features_names = []
        self.features_values = {}
        self.features_values_names = features_values_names
        for i, k in enumerate(features_values_names.keys()):
            self.features_values[i] = len(features_values_names[k])
            self.features_names.append(k)

        self.features = list(range(len(self.features_values)))

    def __entropy(self, data):
        try:
            if len(data) == 0:
                return 0

            # count how many examples are in each class:
            count_classes = [0] * self.classes
            for example in data:
                cls = example[-1]
                count_classes[cls] += 1

            class_probs = [cnt / len(data) for cnt in count_classes]
            # calculate by formula of entropy:
            entropy = 0
            for p_cls in class_probs:
                if p_cls != 0:
                    entropy -= p_cls * log2(p_cls)

            return  entropy
        except:
            print('h')


    def __split_by_feature(self, data, feature):
        feature_values = self.features_values[feature]
        data_by_feature_values = {val:[] for val in range(feature_values)}
        for example in data:
            v = example[feature]
            data_by_feature_values[v].append(example)

        empty = []
        for k in data_by_feature_values:
            if len(data_by_feature_values[k]) == 0:
                empty.append(k)
        for k in empty:
            del (data_by_feature_values[k])

        return data_by_feature_values

    def __calc_information_gain(self, data, feature):
        # calculate entropy of data before splited by attribute.
        entropy_node = self.__entropy(data)

        # split by feature and calculate the entropy after:
        data_splited = self.__split_by_feature(data, feature)
        entropy_childs_weighted = [(len(data_child) / len(data) * self.__entropy(data_child))
                                   for data_child in data_splited.values()]

        ig = entropy_node - sum(entropy_childs_weighted)
        return ig

    def __is_homogenic(self, data):
        # count classes of examples in data, and return True iff the number of unique examples is 1 -
        # all the same class.
        example_classes = [example[-1] for example in data]
        unique_classes = set(example_classes)
        return len(unique_classes) == 1

    def __majority_class(self, data):
        example_classes = [example[-1] for example in data]
        unique_classes = set(example_classes)
        majority_class = max(unique_classes, key=example_classes.count)
        return majority_class


    def __split_recursively(self, data, used_features):

        # mark remaining features.
        features_remaining = [f for f in self.features if f not in used_features]

        # leaf case 1: - homogenic data. classify by the only class.
        if self.__is_homogenic(data):
            cls = data[0][-1]
            return Node((cls, self.classes_names[cls], None), leaf=True)
            #self.tree.append((used_features, data[0][-1]))

        # leaf case 2: if no feature remain, pick class by majority of data:
        elif len(features_remaining) == 0:
            most_common_class = self.__majority_class(data)
            return Node((most_common_class, self.classes_names[most_common_class], None), leaf=True)
        
        # call recursive:
        else:

            # calc ig for each feature not used yet
            igs = {feature:self.__calc_information_gain(data, feature) for feature in features_remaining}
            # take feature with max ig.
            max_ig_feature = max(igs.items(), key=operator.itemgetter(1))[0]

            # split by feature with max ig, and recursivly split:
            sub_datas = self.__split_by_feature(data, max_ig_feature)
            most_common_class = self.__majority_class(data)

            max_ig_feature_dsc = self.features_names[max_ig_feature]
            nd = Node((max_ig_feature,max_ig_feature_dsc, self.features_values_names[max_ig_feature_dsc]))

            for val in range(self.features_values[max_ig_feature]):
                used_features_sub = used_features.copy()
                used_features_sub[max_ig_feature] = val   
                if val in sub_datas:
                    sub_data = sub_datas[val]
                    child_nd = self.__split_recursively(sub_data, used_features_sub)

                else:
                    #self.tree.append((used_features_sub, most_common_class))
                    child_nd = Node((most_common_class, self.classes_names[most_common_class], None), leaf=True)

                nd.add_branch(val, child_nd)
            
            return nd

    def train(self, train_data):
        self.root_node = self.__split_recursively(train_data, {})
        

    def predict(self, test_example):
        # traverse the tree by attribute values until a leaf - which have the class:
        node = self.root_node
        while not node.is_leaf():
            feature_val = test_example[node.val()]
            node = node.get_branch(feature_val)
        
        cls = node.val()
        return cls

class TesterValidator:

    def __init__(self, train_data):
        self.train_data = train_data

    def kfold_cross_validate(self, model, K):
        """
        k fold cross validation on train data, with model.predict.
        return the average of correctness.
        """
        random.shuffle(self.train_data)
        s = len(self.train_data) // K
        correct_percentages = []
        for i in range(K):

            # devide to train,test subsets by i position.
            train_subset = self.train_data[:i * s] + self.train_data[(i + 1) * s:]
            test_subset = self.train_data[i * s:(i + 1) * s]
            test_subset_x = [val[:-1] for val in test_subset]
            test_subset_y = [val[-1] for val in test_subset]

            # train model on train_subset
            model.train(train_subset)
            # count success on test_subsets.
            count = 0
            for example_x, example_y in zip(test_subset_x, test_subset_y):
                y_hat = model.predict(example_x)
                if example_y == y_hat:
                    count += 1

            # add correct percentage to list of those.
            correct_percentages.append(count / len(test_subset))
        return sum(correct_percentages)/len(correct_percentages)

    def test(self, model, test_data):
        """
        train model on data set, and return test_data predictions accuracy.
        """
        model.train(self.train_data)

        count = 0
        for test_example in test_data:
            y = test_example[-1]
            y_hat = model.predict(test_example)
            if y == y_hat:
                count += 1

        return count / len(test_data)


import help
import algorithms
import random
import numpy as np

if __name__=="__main__":
    # extract train data.
    dh = help.DataHandler()
    dh.load_metadata('attributes')
    dh.load_data('dataset.txt')

    train_data = dh.values
    del(dh.attributes_params['can_eat'])

    # build models.
    knn_model = algorithms.KNN_Model(k=5)
    nb_model = algorithms.Naive_Bayes_Model(dh.classes, [len(v) for v in dh.attributes_params.values()])
    id3_model = algorithms.Decision_Tree_Model(dh.classes, dh.attributes_params)
    
    # K-cross validation:
    accuracies = {}
    validator = algorithms.TesterValidator(train_data)
    accuracies['knn'] = validator.kfold_cross_validate(knn_model, K=5)
    accuracies['naive_bayes'] = validator.kfold_cross_validate(nb_model, K=5)
    accuracies['decision tree'] = validator.kfold_cross_validate(id3_model, K=5)

    print('kfold validation:')
    print(accuracies)

    test_dh = help.DataHandler()
    test_dh.load_metadata('attributes')
    test_dh.load_data('test.txt')

    test_data = test_dh.values

    validator = algorithms.TesterValidator(train_data)
    accuracies['knn'] = validator.test(knn_model, test_data)
    accuracies['naive_bayes'] = validator.test(nb_model, test_data)
    accuracies['decision tree'] = validator.test(id3_model, test_data)

    print('test:')
    print(accuracies)

    # prints the tree:

    rootNode = id3_model.root_node
    tree = rootNode.str()
    print(tree)

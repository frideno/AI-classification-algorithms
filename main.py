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

    # build models.
    #model = algorithms.KNN_Model(k=5)
    #model = algorithms.Naive_Bayes_Model(dh.classes, [len(v) for v in dh.attributes_params.values()][:-1])
    model = algorithms.Decision_Tree(len(dh.classes), [len(v) for v in dh.attributes_params.values()][:-1])

    
    # K-cross validation:
    validator = algorithms.TesterValidator(train_data)
    p = validator.kfold_cross_validate(model, K=10)
    print(p)
    test_examples = [
        [0]*22,
        [0]*21 + [1],
        [1] * 22
    ]
    res = validator.test(model, test_examples)
    for i, branch in enumerate(model.tree):
        print(i, branch)
    print(res)
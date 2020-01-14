import help
import algorithms

if __name__=="__main__":
    # extract train data.
    dh = help.DataHandler()
    dh.load_metadata('attributes')
    dh.load_data('train.txt')

    train_data = dh.values
    del(dh.attributes_params['can_eat'])

    # build models.
    knn_model = algorithms.KNN_Model(k=5)
    nb_model = algorithms.Naive_Bayes_Model(dh.classes, [len(v) for v in dh.attributes_params.values()])
    id3_model = algorithms.Decision_Tree_Model(dh.classes, dh.attributes_params)

    test_dh = help.DataHandler()
    test_dh.load_metadata('attributes')
    test_dh.load_data('test.txt')

    test_data = test_dh.values

    accuracies = {}
    validator = algorithms.TesterValidator(train_data)
    accuracies['knn'] = validator.test(knn_model, test_data)
    accuracies['naive_bayes'] = validator.test(nb_model, test_data)
    accuracies['decision tree'] = validator.test(id3_model, test_data)

    # prints the tree:
    rootNode = id3_model.root_node
    with open('output.txt', 'w') as out:
        out.write(rootNode.str()[1:])
        out.write('\n')
        out.write(str(accuracies['decision tree']) + '\t' + str(accuracies['knn']) + '\t' + str(accuracies['naive_bayes']))


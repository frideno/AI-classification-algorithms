import help
import algorithms

if __name__=="__main__":
    # extract train data.
    dh = help.DataHandler()
    dh.load_metadata('dataset.txt')
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


    with open('accuracy.txt', 'w') as f:
        f.write(str(accuracies['decision tree']) + '\t' + str(accuracies['knn']) + '\t' + str(accuracies['naive_bayes']))

    # prints the tree:
    id3_model.train(train_data)
    rootNode = id3_model.root_node
    with open('tree.txt' ,'w') as t:
        t.write(rootNode.str()[1:])




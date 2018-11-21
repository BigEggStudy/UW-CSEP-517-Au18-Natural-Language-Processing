from perceptron import StructuredPerceptron

import utils

if __name__=="__main__":
    #####################################################################

    # small_test_data = list(utils.load_file('..\\data\\test.small'))
    # small_test_data_x = [ input for input, ner_tag in small_test_data ]
    # small_test_data_y = [ ner_tag for input, ner_tag in small_test_data ]

    # print('============= Structure Perceptron (Test) =============')
    # sp = StructuredPerceptron()

    # print('Training Structure Perceptron with iteration 1, and learning rate 0.2')
    # sp.fit(small_test_data, iterations=1, learning_rate=0.2)

    # print('Predict the Dev Set')
    # predict = sp.predict(small_test_data_x)
    # correct, total = utils.predict_eval(small_test_data_y, predict, True)
    # print('Accuracy is %.4f%%, total tags are %d, matched tags are %d' % (correct / total * 100.0, total, correct))
    # print('Output the predict for Evaluation')
    # utils.write_file('..\\output\\unit_test.output', small_test_data, predict)

    #####################################################################

    print('============= Structure Perceptron (Small) =============')
    print('Read Documents and Featurize')
    train_data = list(utils.load_file('..\\data\\eng.train.small'))
    dev_data = list(utils.load_file('..\\data\\eng.dev.small'))
    test_data = list(utils.load_file('..\\data\\eng.test.small'))

    train_data_x = [ input for input, ner_tag in train_data ]
    train_data_y = [ ner_tag for input, ner_tag in train_data ]
    dev_data_x = [ input for input, ner_tag in dev_data ]
    dev_data_y = [ ner_tag for input, ner_tag in dev_data ]
    test_data_x = [ input for input, ner_tag in test_data ]
    test_data_y = [ ner_tag for input, ner_tag in test_data ]

    sp = StructuredPerceptron()

    best_iteration = 0
    best_dev_correction = 0
    # for iteration in [5, 10, 15, 20]:
    for iteration in [5, 10, 15, 20, 50, 75, 100, 150, 200, 250]:
        print(f'Training Structure Perceptron with iteration {iteration}, and learning rate 1')
        sp.fit(train_data, iterations=iteration)

        # print('Predict the Training Set')
        # predict = sp.predict(train_data_x)
        # correct, total = utils.predict_eval(train_data_y, predict)
        # print('Accuracy is %.4f%%, total tags are %d, matched tags are %d' % (correct / total * 100.0, total, correct))

        print('Predict the Dev Set')
        predict = sp.predict(dev_data_x)
        correct, total = utils.predict_eval(dev_data_y, predict)
        print('Accuracy is %.4f%%, total tags are %d, matched tags are %d' % (correct / total * 100.0, total, correct))
        print('Output the predict for Evaluation')
        utils.write_file(f'..\\output\\dev.small.output{iteration}', dev_data, predict)
        if best_dev_correction < correct:
            best_dev_correction = correct
            best_iteration = iteration

        print('Predict the Test Set')
        predict = sp.predict(test_data_x)
        correct, total = utils.predict_eval(test_data_y, predict)
        print('Accuracy is %.4f%%, total tags are %d, matched tags are %d' % (correct / total * 100.0, total, correct))
        print('Output the predict for Evaluation')
        utils.write_file(f'..\\output\\test.small.output{iteration}', test_data, predict)

    print(f'When training with {best_iteration} iteration, Dev Set have highest accuracy')

    #####################################################################

    # print('============= Structure Perceptron (Full Set) =============')
    # best_iteration = 75
    # print('Read Documents and Featurize')
    # train_data = list(utils.load_file('..\\data\\eng.train'))
    # dev_data = list(utils.load_file('..\\data\\eng.dev'))
    # test_data = list(utils.load_file('..\\data\\eng.test'))

    # dev_data_x = [ input for input, ner_tag in dev_data ]
    # dev_data_y = [ ner_tag for input, ner_tag in dev_data ]
    # test_data_x = [ input for input, ner_tag in test_data ]
    # test_data_y = [ ner_tag for input, ner_tag in test_data ]

    # sp = StructuredPerceptron()

    # for iteration in [best_iteration - 25, best_iteration, best_iteration + 25]:
    #     print(f'Training Structure Perceptron with iteration {iteration}, and learning rate 1')
    #     sp.fit(train_data, iterations=iteration)

    #     print('Predict the Dev Set')
    #     predict = sp.predict(dev_data_x)
    #     correct, total = utils.predict_eval(dev_data_y, predict)
    #     print('Accuracy is %.4f%%, total tags are %d, matched tags are %d' % (correct / total * 100.0, total, correct))
    #     print('Output the predict for Evaluation')
    #     utils.write_file(f'..\\output\\dev.output{iteration}', dev_data, predict)

    #     print('Predict the Test Set')
    #     predict = sp.predict(test_data_x)
    #     correct, total = utils.predict_eval(test_data_y, predict)
    #     print('Accuracy is %.4f%%, total tags are %d, matched tags are %d' % (correct / total * 100.0, total, correct))
    #     print('Output the predict for Evaluation')
    #     utils.write_file(f'..\\output\\test.output{iteration}', test_data, predict)

    #####################################################################

    print('============= Structure Perceptron (Ablation) =============')
    print('Read Documents and Featurize')
    train_data = list(utils.load_file('..\\data\\eng.train.small'))
    dev_data = list(utils.load_file('..\\data\\eng.dev.small'))
    test_data = list(utils.load_file('..\\data\\eng.test.small'))

    train_data_x = [ input for input, ner_tag in train_data ]
    train_data_y = [ ner_tag for input, ner_tag in train_data ]
    dev_data_x = [ input for input, ner_tag in dev_data ]
    dev_data_y = [ ner_tag for input, ner_tag in dev_data ]
    test_data_x = [ input for input, ner_tag in test_data ]
    test_data_y = [ ner_tag for input, ner_tag in test_data ]

    sp = StructuredPerceptron()
    for ablation in range(1, 5):
        print(f'Training Structure Perceptron with iteration 75, and learning rate 1')
        sp.fit(train_data, iterations=75, feature_test=ablation)

        print('Predict the Dev Set')
        predict = sp.predict(dev_data_x, ablation)
        correct, total = utils.predict_eval(dev_data_y, predict)
        print('Accuracy is %.4f%%, total tags are %d, matched tags are %d' % (correct / total * 100.0, total, correct))
        print('Output the predict for Evaluation')
        utils.write_file(f'..\\output\\dev.small.ablation_{ablation}', dev_data, predict)

        print('Predict the Test Set')
        predict = sp.predict(test_data_x, ablation)
        correct, total = utils.predict_eval(test_data_y, predict)
        print('Accuracy is %.4f%%, total tags are %d, matched tags are %d' % (correct / total * 100.0, total, correct))
        print('Output the predict for Evaluation')
        utils.write_file(f'..\\output\\test.small.ablation_{ablation}', test_data, predict)

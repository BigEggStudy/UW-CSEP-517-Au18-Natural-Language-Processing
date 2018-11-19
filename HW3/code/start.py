from perceptron import StructuredPerceptron

import utils

if __name__=="__main__":
    #####################################################################

    small_test_data = list(utils.load_file('..\\data\\test.small'))
    small_test_data_x = [ input for input, ner_tag in small_test_data ]
    small_test_data_y = [ ner_tag for input, ner_tag in small_test_data ]

    print('============= Structure Preceptron (Test) =============')
    sp = StructuredPerceptron()

    print('Training Struecture Preceptron with iteration 1, and learning rate 0.2')
    sp.fit(small_test_data, iterations=1, learning_rate=0.2)

    print('Predict the Dev Set')
    predict = sp.predict(small_test_data_x)
    correct, total = utils.predict_eval(small_test_data_y, predict, True)
    print('Accuracy is %.4f%%, total tags are %d, matched tags are %d' % (correct / total, total, correct))
    print('Output the predict for Evaluation')
    utils.write_file('.\dev.small.output', small_test_data, predict)

    #####################################################################

    # print('Read Documents and Featurize')
    # train_data = list(utils.load_file('..\\data\\eng.train.small'))
    # dev_data = list(utils.load_file('..\\data\\eng.dev.small'))
    # test_data = list(utils.load_file('..\\data\\eng.test.small'))

    # dev_data_x = [ input for input, ner_tag in dev_data ]
    # dev_data_y = [ ner_tag for input, ner_tag in dev_data ]
    # test_data_x = [ input for input, ner_tag in test_data ]
    # test_data_y = [ ner_tag for input, ner_tag in test_data ]

    # print('============= Structure Preceptron (Small) =============')
    # sp = StructuredPerceptron()

    # print('Training Struecture Preceptron with iteration 1, and learning rate 0.2')
    # sp.fit(train_data, iterations=1, learning_rate=0.2)

    # print('Predict the Dev Set')
    # predict = sp.predict(dev_data_x)
    # correct, total = utils.predict_eval(dev_data_y, predict)
    # print('Accuracy is %.4f%%, total tags are %d, matched tags are %d' % (correct / total, total, correct))
    # print('Output the predict for Evaluation')
    # utils.write_file('.\dev.small.output', dev_data, predict)

    # print('Predict the Test Set')
    # predict = sp.predict(test_data_x)
    # correct, total = utils.predict_eval(test_data_y, predict)
    # print('Accuracy is %.4f%%, total tags are %d, matched tags are %d' % (correct / total, total, correct))
    # print('Output the predict for Evaluation')
    # utils.write_file('.\test.small.output', test_data, predict)

import AssignmentHelper
import MeasureAccuracy
import Constant

if __name__ == '__main__':
    data_raw_train = AssignmentHelper.read_sentences_from_file('..\\data\\twt.train.json')
    data_raw_dev = AssignmentHelper.read_sentences_from_file('..\\data\\twt.dev.json')
    data_raw_test = AssignmentHelper.read_sentences_from_file('..\\data\\twt.test.json')

    (data_dev_sentences, data_dev_labels) = AssignmentHelper.extract_labels(data_raw_dev)
    (data_test_sentences, data_test_labels) = AssignmentHelper.extract_labels(data_raw_test)

    for sentence in data_dev_sentences:
        sentence.insert(0, Constant.SENTENCE_START2)
        sentence.insert(0, Constant.SENTENCE_START1)
        sentence.append(Constant.SENTENCE_END)
    for sentence in data_test_sentences:
        sentence.insert(0, Constant.SENTENCE_START2)
        sentence.insert(0, Constant.SENTENCE_START1)
        sentence.append(Constant.SENTENCE_END)

    # print('============= Trigram HMM with UNK =============')
    # import TrigramHMM
    # trigramHMM = TrigramHMM.TrigramHMM()

    # best_unk_threshold = 0
    # best_accuracy = 0
    # for unk_threshold in [1, 3, 5]:
    #     print('Building Trigram HMM with Unknown Threshold as %d' % unk_threshold)
    #     trigramHMM.fit(data_raw_train, unk_threshold)

    #     print('Predict the Dev Dataset')
    #     predict_labels = trigramHMM.predict(data_dev_sentences)
    #     (totalmatch, totalcount, accuracy) = MeasureAccuracy.calculate_accuracy(data_dev_labels, predict_labels)
    #     if accuracy > best_accuracy:
    #         best_accuracy = accuracy
    #         best_unk_threshold = unk_threshold
    #     print("Accuracy is %.4f%%, total tags are %d, matched tags are %d" % (accuracy, totalcount, totalmatch))

    # print('When UNK Threshold as %d, the accuracy is the best, use it for other testing' % best_unk_threshold)
    # unk_threshold = best_unk_threshold

    print('============= Trigram HMM with UNK and Add-K Smoothing =============')
    import TrigramHMMWithAddK

    best_accuracy = 0
    best_k1 = 0
    best_k2 = 0
    best_unk_threshold = 0
    for unk_threshold in [1, 3, 5]:
        for k1 in [0.0001, 0.001, 0.01]:
            for k2 in [0.0001, 0.001, 0.01, 0.1, 1, 10]:
                trigramHMM = TrigramHMMWithAddK.TrigramHMM(k1, k2)

                print('Building Trigram HMM with Unknown Threshold as %d, and k (for emission) as %f, k (for transition) as %f' % (unk_threshold, k1, k2))
                trigramHMM.fit(data_raw_train, unk_threshold)

                print('Predict the Dev Dataset')
                predict_labels = trigramHMM.predict(data_dev_sentences)
                (totalmatch, totalcount, accuracy) = MeasureAccuracy.calculate_accuracy(data_dev_labels, predict_labels)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_unk_threshold = unk_threshold
                    best_k1 = k1
                    best_k2 = k2
                print("Accuracy is %.4f%%, total tags are %d, matched tags are %d" % (accuracy, totalcount, totalmatch))

    print('When UNK Threshold as %d, k (for emission) as %f, k (for transition) as %f, the accuracy is the best, use it for other testing' % (best_unk_threshold, best_k1, best_k2))
    k1 = best_k1
    k2 = best_k2
    unk_threshold = best_unk_threshold

    unk_threshold = 1
    k1 = 0.0001
    k2 = 0.1

    print('============= Trigram HMM with UNK and Linear Interpolation Smoothing =============')
    import TrigramHMMWithLI

    best_accuracy = 0
    best_l1 = 0
    best_l2 = 0
    best_l3 = 0
    for (l1, l2) in [(0.5, 0.25), (0.333, 0.333), (0.667, 0.1), (0.75, 0.1), (0.25, 0.5)]:
        l3 = 1 - (l1 + l2)
        trigramHMM = TrigramHMMWithLI.TrigramHMM(l1, l2, l3)

        print('Building Trigram HMM with Unknown Threshold as %d, and lambda as %f, %f, %f' % (unk_threshold, l1, l2, l3))
        trigramHMM.fit(data_raw_train, unk_threshold)

        print('Predict the Dev Dataset')
        predict_labels = trigramHMM.predict(data_dev_sentences)
        (totalmatch, totalcount, accuracy) = MeasureAccuracy.calculate_accuracy(data_dev_labels, predict_labels)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_l1 = l1
            best_l2 = l2
            best_l3 = l3
        print("Accuracy is %.4f%%, total tags are %d, matched tags are %d" % (accuracy, totalcount, totalmatch))

    print('When lambda as %f, %f, %f, the accuracy is the best, use it for other testing' % (best_l1, best_l2, best_l3))
    l1 = best_l1
    l2 = best_l2
    l3 = best_l3

    print('============= Trigram HMM with UNK, Add-K and Linear Interpolation Smoothing =============')
    import TrigramHMMWithSmoothing

    best_accuracy = 0
    best_l1 = 0
    best_l2 = 0
    best_l3 = 0
    for lambda1 in range(5, 100, 5):
        for lambda2 in range(5, 100, 5):
            if lambda1 + lambda2 >= 100:
                break
            l1 = float(lambda1) / 100
            l2 = float(lambda2) / 100
            l3 = 1 - (l1 + l2)
            trigramHMM = TrigramHMMWithSmoothing.TrigramHMM(k1, k2, l1, l2, l3)

            print('Building Trigram HMM with Unknown Threshold as %d, k (for emission) as %f, k (for transition) as %f, and lambda as %f, %f, %f' % (unk_threshold, k1, k2, l1, l2, l3))
            trigramHMM.fit(data_raw_train, unk_threshold)

            print('Predict the Dev Dataset')
            predict_labels = trigramHMM.predict(data_dev_sentences)
            (totalmatch, totalcount, accuracy) = MeasureAccuracy.calculate_accuracy(data_dev_labels, predict_labels)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_l1 = l1
                best_l2 = l2
                best_l3 = l3
            print("Accuracy is %.4f%%, total tags are %d, matched tags are %d" % (accuracy, totalcount, totalmatch))

    print('When lambda as %f, %f, %f, the accuracy is the best, use it for other testing' % (best_l1, best_l2, best_l3))
    l1 = best_l1
    l2 = best_l2
    l3 = best_l3

    print('============= Trigram HMM with Best Parameters =============')
    import TrigramHMMWithSmoothing
    trigramHMM = TrigramHMMWithSmoothing.TrigramHMM(k1, k2, l1, l2, l3)

    print('Building Trigram HMM with Unknown Threshold as %d, k (for emission) as %f, k (for transition) as %f, and lambda as %f, %f, %f' % (unk_threshold, k1, k2, l1, l2, l3))
    trigramHMM.fit(data_raw_train, unk_threshold)

    print('Predict the Dev Dataset')
    predict_labels = trigramHMM.predict(data_dev_sentences)
    (totalmatch, totalcount, accuracy) = MeasureAccuracy.calculate_accuracy(data_dev_labels, predict_labels)
    print("Accuracy is %.4f%%, total tags are %d, matched tags are %d" % (accuracy, totalcount, totalmatch))
    (confusion_matrix, expected_tags, predicted_tags) = MeasureAccuracy.calculate_confusion_matrix(data_dev_labels, predict_labels)
    error_sentences = MeasureAccuracy.get_most_error_data_trigram(data_dev_labels, predict_labels, data_dev_sentences)

    print('Predict the Test Dataset')
    predict_labels = trigramHMM.predict(data_test_sentences)
    (totalmatch, totalcount, accuracy) = MeasureAccuracy.calculate_accuracy(data_test_labels, predict_labels)
    print("Accuracy is %.4f%%, total tags are %d, matched tags are %d" % (accuracy, totalcount, totalmatch))

    print('============= Error analysis for Dev Dataset =============')
    print('The confusion matrix:')
    sorted(expected_tags)
    sorted(predicted_tags)

    header1 = '     '
    header2 = '-----'
    for predict_tag in predicted_tags:
        header1 += ' |   ' + predict_tag + '  '
        header2 += ' | -----'
    print(header1)
    print(header2)

    for expected_tag in expected_tags:
        line = '  ' + expected_tag + '  '
        for predict_tag in predicted_tags:
             line += ' | ' + '{:>5}'.format(confusion_matrix[expected_tag][predict_tag])
        print(line)

    print('Top 10 data contain most errors:')
    for (sentence, expected, predicted) in error_sentences:
        print(sentence)
        print(expected)
        print(predicted)

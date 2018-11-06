import AssignmentHelper

if __name__ == '__main__':
    data_raw_train = AssignmentHelper.read_sentences_from_file('..\\data\\brown.train.txt')
    data_raw_dev = AssignmentHelper.read_sentences_from_file('..\\data\\brown.dev.txt')
    data_raw_test = AssignmentHelper.read_sentences_from_file('..\\data\\brown.test.txt')

    # print('============= Add-K Smoothing =============')
    # import TrigramLMWithAddK

    # data_train_3 = AssignmentHelper.prepare_data_trigram(data_raw_train)
    # data_dev_3 = AssignmentHelper.prepare_data_trigram(data_raw_dev)
    # data_test_3 = AssignmentHelper.prepare_data_trigram(data_raw_test)

    # for unk_threshold in [0, 1, 3, 5]:
    #     for k in [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]:
    #         print('Building Trigram Language Model with Unknown Threshold as %d, and k as %f' % (unk_threshold, k))
    #         trigramLMWithAddK = TrigramLMWithAddK.TrigramLMWithAddK(data_train_3, k, unk_threshold)

    #         print('Validate the Trigram Language Model')
    #         (validation, probability_sum, word1, word2) = trigramLMWithAddK.validate()
    #         if not validation:
    #             print('Building Trigram Language Model FAILED, word pair "%s, %s" total probability is %.4f' % (word1, word2, probability_sum))
    #             exit()
    #         else:
    #             print('Building Trigram Language Model SUCCESS')

    #         print('Compute the Preplexity of Trigram Language Model based on Train Data')
    #         (preplexity, zero_prob_sentence_persentage, zero_prob_sentence_count) = trigramLMWithAddK.calculate_perplexity(data_train_3)
    #         print("%.0f%% (%d) of Zero Prob Sentences, Perplexity on Other Sentences is %s" % (zero_prob_sentence_persentage, zero_prob_sentence_count, preplexity))
    #         print('Compute the Preplexity of Trigram Language Model based on Dev Data')
    #         (preplexity, zero_prob_sentence_persentage, zero_prob_sentence_count) = trigramLMWithAddK.calculate_perplexity(data_dev_3)
    #         print("%.0f%% (%d) of Zero Prob Sentences, Perplexity on Other Sentences is %s" % (zero_prob_sentence_persentage, zero_prob_sentence_count, preplexity))

    print('============= Linear Interpolation Smoothing =============')
    import TrigramLMWithLI

    data_train_3 = AssignmentHelper.prepare_data_trigram(data_raw_train)
    data_dev_3 = AssignmentHelper.prepare_data_trigram(data_raw_dev)
    data_test_3 = AssignmentHelper.prepare_data_trigram(data_raw_test)

    for unk_threshold in [0, 1, 3, 5]:
        for (l1, l2) in [(0.5, 0.25), (0.333, 0.333), (0.667, 0.1), (0.75, 0.1), (0.25, 0.5)]:
            l3 = 1 - (l1 + l2)

            print('Building Trigram Language Model with Unknown Threshold as %d, and lambda as %f, %f, %f' % (unk_threshold, l1, l2, l3))
            trigramLMWithLI = TrigramLMWithLI.TrigramLMWithLI(data_train_3, l1, l2, l3, unk_threshold)

            print('Validate the Trigram Language Model')
            (validation, probability_sum, word1, word2) = trigramLMWithLI.validate()
            if not validation:
                print('Building Trigram Language Model FAILED, word pair "%s, %s" total probability is %.4f' % (word1, word2, probability_sum))
                exit()
            else:
                print('Building Trigram Language Model SUCCESS')

            print('Compute the Preplexity of Trigram Language Model based on Train Data')
            (preplexity, zero_prob_sentence_persentage, zero_prob_sentence_count) = trigramLMWithLI.calculate_perplexity(data_train_3)
            print("%.0f%% (%d) of Zero Prob Sentences, Perplexity on Other Sentences is %s" % (zero_prob_sentence_persentage, zero_prob_sentence_count, preplexity))
            print('Compute the Preplexity of Trigram Language Model based on Dev Data')
            (preplexity, zero_prob_sentence_persentage, zero_prob_sentence_count) = trigramLMWithLI.calculate_perplexity(data_dev_3)
            print("%.0f%% (%d) of Zero Prob Sentences, Perplexity on Other Sentences is %s" % (zero_prob_sentence_persentage, zero_prob_sentence_count, preplexity))

    print('============= Add-K & Linear Interpolation Smoothing =============')
    import TrigramLMWithSmoothing

    data_train_3 = AssignmentHelper.prepare_data_trigram(data_raw_train)
    data_dev_3 = AssignmentHelper.prepare_data_trigram(data_raw_dev)
    data_test_3 = AssignmentHelper.prepare_data_trigram(data_raw_test)

    for unk_threshold in [1, 3, 5]:
        k = 0.00001
        l1 = 0.25
        l2 = 0.5
        l3 = 1 - (l1 + l2)

        print('Building Trigram Language Model with Unknown Threshold as %d, k as %f, and lambda as %f, %f, %f' % (unk_threshold, k, l1, l2, l3))
        trigramLMWithSmoothing = TrigramLMWithSmoothing.TrigramLMWithSmoothing(data_train_3, k, l1, l2, l3, unk_threshold)

        print('Validate the Trigram Language Model')
        (validation, probability_sum, word1, word2) = trigramLMWithSmoothing.validate()
        if not validation:
            print('Building Trigram Language Model FAILED, word pair "%s, %s" total probability is %.4f' % (word1, word2, probability_sum))
            exit()
        else:
            print('Building Trigram Language Model SUCCESS')

        print('Compute the Preplexity of Trigram Language Model based on Test Data')
        (preplexity, zero_prob_sentence_persentage, zero_prob_sentence_count) = trigramLMWithSmoothing.calculate_perplexity(data_test_3)
        print("%.0f%% (%d) of Zero Prob Sentences, Perplexity on Other Sentences is %s" % (zero_prob_sentence_persentage, zero_prob_sentence_count, preplexity))

    print('============= Add-K & Linear Interpolation Smoothing & Only Training Half Training Set =============')
    import TrigramLMWithSmoothing

    data_train_3 = AssignmentHelper.prepare_data_trigram(data_raw_train)
    data_dev_3 = AssignmentHelper.prepare_data_trigram(data_raw_dev)
    data_test_3 = AssignmentHelper.prepare_data_trigram(data_raw_test)
    print('Previously the Training Set Size is %d' % len(data_train_3))
    data_train_3 = data_train_3[:round(len(data_train_3) / 2)]
    print('Now the Training Set Size is %d' % len(data_train_3))

    for unk_threshold in [1, 3, 5]:
        k = 0.00001
        l1 = 0.25
        l2 = 0.5
        l3 = 1 - (l1 + l2)

        print('Building Trigram Language Model with Unknown Threshold as %d, k as %f, and lambda as %f, %f, %f' % (unk_threshold, k, l1, l2, l3))
        trigramLMWithSmoothing = TrigramLMWithSmoothing.TrigramLMWithSmoothing(data_train_3, k, l1, l2, l3, unk_threshold)

        print('Validate the Trigram Language Model')
        (validation, probability_sum, word1, word2) = trigramLMWithSmoothing.validate()
        if not validation:
            print('Building Trigram Language Model FAILED, word pair "%s, %s" total probability is %.4f' % (word1, word2, probability_sum))
            exit()
        else:
            print('Building Trigram Language Model SUCCESS')

        print('Compute the Preplexity of Trigram Language Model based on Test Data')
        (preplexity, zero_prob_sentence_persentage, zero_prob_sentence_count) = trigramLMWithSmoothing.calculate_perplexity(data_test_3)
        print("%.0f%% (%d) of Zero Prob Sentences, Perplexity on Other Sentences is %s" % (zero_prob_sentence_persentage, zero_prob_sentence_count, preplexity))

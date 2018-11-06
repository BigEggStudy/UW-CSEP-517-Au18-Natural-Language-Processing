import AssignmentHelper

if __name__ == '__main__':
    data_raw_train = AssignmentHelper.read_sentences_from_file('..\\data\\brown.train.txt')
    data_raw_dev = AssignmentHelper.read_sentences_from_file('..\\data\\brown.dev.txt')
    data_raw_test = AssignmentHelper.read_sentences_from_file('..\\data\\brown.test.txt')

    print('============= Unigram =============')
    import UnigramLM

    data_train_1 = AssignmentHelper.prepare_data_unigram(data_raw_train)
    data_dev_1 = AssignmentHelper.prepare_data_unigram(data_raw_dev)
    data_test_1 = AssignmentHelper.prepare_data_unigram(data_raw_test)

    for unk_threshold in [0, 1, 3, 5, 10, 20, 30, 50]:
        print('Building Unigram Language Model with Unknown Threshold as %d' % unk_threshold)
        unigramLM = UnigramLM.UnigramLM(data_train_1, unk_threshold)

        print('Validate the Language Model')
        (validation, probability_sum) = unigramLM.validate()
        if not validation:
            print('Building Unigram Language Model FAILED, total probability is %.4f' % (probability_sum))
            exit()
        else:
            print('Building Unigram Language Model SUCCESS')

        print('Compute the Preplexity of Unigram Language Model based on Train Data')
        (preplexity, zero_prob_sentence_persentage, zero_prob_sentence_count) = unigramLM.calculate_perplexity(data_train_1)
        print("%.0f%% (%d) of Zero Prob Sentences, Perplexity on Other Sentences is %s" % (zero_prob_sentence_persentage, zero_prob_sentence_count, preplexity))
        print('Compute the Preplexity of Unigram Language Model based on Dev Data')
        (preplexity, zero_prob_sentence_persentage, zero_prob_sentence_count) = unigramLM.calculate_perplexity(data_dev_1)
        print("%.0f%% (%d) of Zero Prob Sentences, Perplexity on Other Sentences is %s" % (zero_prob_sentence_persentage, zero_prob_sentence_count, preplexity))
        print('Compute the Preplexity of Unigram Language Model based on Test Data')
        (preplexity, zero_prob_sentence_persentage, zero_prob_sentence_count) = unigramLM.calculate_perplexity(data_test_1)
        print("%.0f%% (%d) of Zero Prob Sentences, Perplexity on Other Sentences is %s" % (zero_prob_sentence_persentage, zero_prob_sentence_count, preplexity))

    print('============= Bigram =============')
    import BigramLM

    data_train_2 = AssignmentHelper.prepare_data_bigram(data_raw_train)
    data_dev_2 = AssignmentHelper.prepare_data_bigram(data_raw_dev)
    data_test_2 = AssignmentHelper.prepare_data_bigram(data_raw_test)

    for unk_threshold in [0, 1, 3, 5, 10, 20, 30, 50]:
        print('Building Bigram Language Model with Unknown Threshold as %d' % unk_threshold)
        bigramLM = BigramLM.BigramLM(data_train_2, unk_threshold)

        print('Validate the Bigram Language Model')
        (validation, probability_sum, word) = bigramLM.validate()
        if not validation:
            print('Building Bigram Language Model FAILED, word "%s" total probability is %.4f' % (word, probability_sum))
            exit()
        else:
            print('Building Bigram Language Model SUCCESS')

        print('Compute the Preplexity of Bigram Language Model based on Train Data')
        (preplexity, zero_prob_sentence_persentage, zero_prob_sentence_count) = bigramLM.calculate_perplexity(data_train_2)
        print("%.0f%% (%d) of Zero Prob Sentences, Perplexity on Other Sentences is %s" % (zero_prob_sentence_persentage, zero_prob_sentence_count, preplexity))
        print('Compute the Preplexity of Bigram Language Model based on Dev Data')
        (preplexity, zero_prob_sentence_persentage, zero_prob_sentence_count) = bigramLM.calculate_perplexity(data_dev_2)
        print("%.0f%% (%d) of Zero Prob Sentences, Perplexity on Other Sentences is %s" % (zero_prob_sentence_persentage, zero_prob_sentence_count, preplexity))
        print('Compute the Preplexity of Bigram Language Model based on Test Data')
        (preplexity, zero_prob_sentence_persentage, zero_prob_sentence_count) = bigramLM.calculate_perplexity(data_test_2)
        print("%.0f%% (%d) of Zero Prob Sentences, Perplexity on Other Sentences is %s" % (zero_prob_sentence_persentage, zero_prob_sentence_count, preplexity))

    print('============= Trigram =============')
    import TrigramLM

    data_train_3 = AssignmentHelper.prepare_data_trigram(data_raw_train)
    data_dev_3 = AssignmentHelper.prepare_data_trigram(data_raw_dev)
    data_test_3 = AssignmentHelper.prepare_data_trigram(data_raw_test)

    for unk_threshold in [0, 1, 3, 5, 10, 20, 30, 50]:
        print('Building Trigram Language Model with Unknown Threshold as %d' % unk_threshold)
        trigramLM = TrigramLM.TrigramLM(data_train_3, unk_threshold)

        print('Validate the Trigram Language Model')
        (validation, probability_sum, word1, word2) = trigramLM.validate()
        if not validation:
            print('Building Trigram Language Model FAILED, word pair "%s, %s" total probability is %.4f' % (word1, word2, probability_sum))
            exit()
        else:
            print('Building Trigram Language Model SUCCESS')

        print('Compute the Preplexity of Trigram Language Model based on Train Data')
        (preplexity, zero_prob_sentence_persentage, zero_prob_sentence_count) = trigramLM.calculate_perplexity(data_train_3)
        print("%.0f%% (%d) of Zero Prob Sentences, Perplexity on Other Sentences is %s" % (zero_prob_sentence_persentage, zero_prob_sentence_count, preplexity))
        print('Compute the Preplexity of Trigram Language Model based on Dev Data')
        (preplexity, zero_prob_sentence_persentage, zero_prob_sentence_count) = trigramLM.calculate_perplexity(data_dev_3)
        print("%.0f%% (%d) of Zero Prob Sentences, Perplexity on Other Sentences is %s" % (zero_prob_sentence_persentage, zero_prob_sentence_count, preplexity))
        print('Compute the Preplexity of Trigram Language Model based on Test Data')
        (preplexity, zero_prob_sentence_persentage, zero_prob_sentence_count) = trigramLM.calculate_perplexity(data_test_3)
        print("%.0f%% (%d) of Zero Prob Sentences, Perplexity on Other Sentences is %s" % (zero_prob_sentence_persentage, zero_prob_sentence_count, preplexity))

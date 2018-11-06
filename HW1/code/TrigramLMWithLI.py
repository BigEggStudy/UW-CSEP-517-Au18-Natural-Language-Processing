import math
import sys
import BigramLM
import Constant

class TrigramLMWithLI(BigramLM.BigramLM):
    def __init__(self, sentences, l1, l2, l3, unk_threshold):
        BigramLM.BigramLM.__init__(self, sentences, unk_threshold)

        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.trigram_frequencies = dict()
        for sentence in sentences:
            keys = zip(sentence, sentence[1:], sentence[2:])

            for key in keys:
                (key, updated) = self.__update_key(key[0], key[1], key[2])
                if (key[0], key[1]) not in self.trigram_frequencies:
                    self.trigram_frequencies[(key[0], key[1])] = dict()
                self.trigram_frequencies[(key[0], key[1])][key[2]] = self.trigram_frequencies[(key[0], key[1])].get(key[2], 0) + 1

    def validate(self):
        for key in self.trigram_frequencies.keys():
            probability_sum = 0
            for word3 in self.trigram_frequencies[key].keys():
                probability_sum += self.__calculate_trigram_probability(key[0], key[1], word3)
            probability_sum += self.__calculate_trigram_probability(key[0], key[1], Constant.UNK)

            if probability_sum <= 0.99 and probability_sum >= 1.01:
                return (False, probability_sum, key[0], key[1])

        return (True, 1.00, '', '')

    def __update_key(self, key0, key1, key2):
        newKey0 = key0
        newKey1 = key1
        newKey2 = key2
        updated = False
        if self.unk_threshold > 0:
            if key0 not in self.vocabulary or key0 in self.unk_unigram_keys:
                newKey0 = Constant.UNK
                updated = True
            if key1 not in self.vocabulary or key1 in self.unk_unigram_keys:
                newKey1 = Constant.UNK
                updated = True
            if key2 not in self.vocabulary or key2 in self.unk_unigram_keys:
                newKey2 = Constant.UNK
                updated = True
        return ((newKey0, newKey1, newKey2), updated)

    def calculate_perplexity(self, sentences):
        sentence_probability_log_sum = 0
        zero_prob_sentence_count = 0
        m = 0
        for sentence in sentences:
            m += len(sentence)
            keys = zip(sentence, sentence[1:], sentence[2:])

            for key in keys:
                (key, updated) = self.__update_key(key[0], key[1], key[2])

                trigram_word_probability = self.l1 * self.__calculate_trigram_probability(key[0], key[1], key[2]) + \
                                           self.l2 * self.__calculate_bigram_probability(key[1], key[2]) + \
                                           self.l3 * self.__calculate_unigram_probability(key[2])
                if trigram_word_probability == 0.0:
                    zero_prob_sentence_count += 1
                    m -= len(sentence)
                    break
                sentence_probability_log_sum += math.log2(trigram_word_probability)

        l = sentence_probability_log_sum / m

        zero_prob_sentence_persentage = zero_prob_sentence_count / len(sentences) * 100
        try:
            return (math.pow(2, -l), zero_prob_sentence_persentage, zero_prob_sentence_count)
        except:
            return (sys.maxsize, zero_prob_sentence_persentage, zero_prob_sentence_count)

    def __calculate_trigram_probability(self, previous_previous_word, previous_word, word):
        if (previous_previous_word, previous_word) not in self.trigram_frequencies:
            return 0.0

        word_probability_denominator = sum(self.trigram_frequencies[(previous_previous_word, previous_word)].values())
        word_probability_numerator = self.trigram_frequencies[(previous_previous_word, previous_word)].get(word, 0)

        if word_probability_numerator == 0 or word_probability_denominator == 0:
            return 0.0
        else:
            return float(word_probability_numerator) / float(word_probability_denominator)

    def __calculate_bigram_probability(self, previous_word, word):
        word_probability_denominator = sum(self.bigram_frequencies[previous_word].values())
        word_probability_numerator = self.bigram_frequencies[previous_word].get(word, 0)

        if word_probability_numerator == 0 or word_probability_denominator == 0:
            return 0.0
        else:
            return float(word_probability_numerator) / float(word_probability_denominator)

    def __calculate_unigram_probability(self, word):
        word_probability_numerator = self.unigram_frequencies.get(word, 0)
        word_probability_denominator = self.corpus_length
        return float(word_probability_numerator) / float(word_probability_denominator)

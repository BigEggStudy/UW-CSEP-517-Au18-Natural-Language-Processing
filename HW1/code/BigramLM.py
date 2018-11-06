import math
import sys
import UnigramLM
import Constant

class BigramLM(UnigramLM.UnigramLM):
    def __init__(self, sentences, unk_threshold):
        UnigramLM.UnigramLM.__init__(self, sentences, unk_threshold)

        self.bigram_frequencies = dict()
        for sentence in sentences:
            keys = zip(sentence, sentence[1:])

            for key in keys:
                (key, updated) = self.__update_key(key[0], key[1])
                if key[0] not in self.bigram_frequencies:
                    self.bigram_frequencies[key[0]] = dict()
                self.bigram_frequencies[key[0]][key[1]] = self.bigram_frequencies[key[0]].get(key[1], 0) + 1

    def validate(self):
        for word1 in self.bigram_frequencies.keys():
            probability_sum = 0
            for word2 in self.bigram_frequencies[word1].keys():
                probability_sum += self.__calculate_bigram_probability(word1, word2)
            probability_sum += self.__calculate_bigram_probability(word1, Constant.UNK)

            if probability_sum <= 0.99 and probability_sum >= 1.01:
                return (False, probability_sum, word1)

        return (True, 1.00, '')

    def __update_key(self, key0, key1):
        newKey0 = key0
        newKey1 = key1
        updated = False
        if self.unk_threshold > 0:
            if key0 not in self.vocabulary or key0 in self.unk_unigram_keys:
                newKey0 = Constant.UNK
                updated = True
            if key1 not in self.vocabulary or key1 in self.unk_unigram_keys:
                newKey1 = Constant.UNK
                updated = True
        return ((newKey0, newKey1), updated)

    def calculate_perplexity(self, sentences):
        sentence_probability_log_sum = 0
        zero_prob_sentence_count = 0
        m = 0
        for sentence in sentences:
            keys = zip(sentence, sentence[1:])
            m += len(sentence)

            for key in keys:
                (key, updated) = self.__update_key(key[0], key[1])

                bigram_word_probability = self.__calculate_bigram_probability(key[0], key[1])
                if bigram_word_probability == 0.0:
                    zero_prob_sentence_count += 1
                    m -= len(sentence)
                    break
                sentence_probability_log_sum += math.log2(bigram_word_probability)

        l = sentence_probability_log_sum / m

        zero_prob_sentence_persentage = zero_prob_sentence_count / len(sentences) * 100
        try:
            return (math.pow(2, -l), zero_prob_sentence_persentage, zero_prob_sentence_count)
        except:
            return (sys.maxsize, zero_prob_sentence_persentage, zero_prob_sentence_count)

    def __calculate_bigram_probability(self, previous_word, word):
        word_probability_denominator = sum(self.bigram_frequencies[previous_word].values())
        word_probability_numerator = self.bigram_frequencies[previous_word].get(word, 0)

        if word_probability_numerator == 0 or word_probability_denominator == 0:
            return 0.0
        else:
            return float(word_probability_numerator) / float(word_probability_denominator)

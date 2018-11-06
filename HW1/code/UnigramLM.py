import math
import sys
import Constant

class UnigramLM:
    def __init__(self, sentences, unk_threshold):
        self.unk_threshold = unk_threshold
        self.unigram_frequencies = dict()
        self.corpus_length = 0
        self.vocabulary = set()
        for sentence in sentences:
            for word in sentence:
                self.unigram_frequencies[word] = self.unigram_frequencies.get(word, 0) + 1
                self.corpus_length += 1
                if word not in self.vocabulary:
                    self.vocabulary.add(word)

        if unk_threshold > 0:
            self.vocabulary.add(Constant.UNK)
            unk_count = 0
            self.unk_unigram_keys = set()
            keys = self.unigram_frequencies.keys()
            for key in keys:
                if self.unigram_frequencies.get(key) <= unk_threshold:
                    unk_count += self.unigram_frequencies.get(key)
                    self.unk_unigram_keys.add(key)

            self.unigram_frequencies[Constant.UNK] = unk_count
            for key in self.unk_unigram_keys:
                del self.unigram_frequencies[key]

    def validate(self):
        probability_sum = 0
        for word in self.vocabulary:
            probability_sum += self.__calculate_unigram_probability(word)

        return (probability_sum > 0.99 and probability_sum < 1.01, probability_sum)

    def calculate_perplexity(self, sentences):
        sentence_probability_log_sum = 0
        zero_prob_sentence_count = 0
        m = 0
        for sentence in sentences:
            m += len(sentence)
            for word in sentence:
                if self.unk_threshold > 0 and (word not in self.vocabulary or word in self.unk_unigram_keys):
                    word = Constant.UNK

                word_probability = self.__calculate_unigram_probability(word)
                if word_probability == 0.0:
                    zero_prob_sentence_count += 1
                    m -= len(sentence)
                    break
                sentence_probability_log_sum += math.log2(word_probability)

        l = sentence_probability_log_sum / m

        zero_prob_sentence_persentage = zero_prob_sentence_count / len(sentences) * 100
        try:
            return (math.pow(2, -l), zero_prob_sentence_persentage, zero_prob_sentence_count)
        except:
            return (sys.maxsize, zero_prob_sentence_persentage, zero_prob_sentence_count)

    def __calculate_unigram_probability(self, word):
        word_probability_numerator = self.unigram_frequencies.get(word, 0)
        word_probability_denominator = self.corpus_length
        return float(word_probability_numerator) / float(word_probability_denominator)

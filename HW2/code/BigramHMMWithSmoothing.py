import math
import sys
from collections import defaultdict
import Constant

class BigramHMM:
    def __init__(self, k1, k2, l1, l2):
        self.l1 = l1
        self.l2 = l2
        self.k1 = k1
        self.k2 = k2

    def fit(self, tagged_sentences, unk_threshold = 1):
        self.tag_frequencies = defaultdict(lambda: 0)
        self.word_frequencies = defaultdict(lambda: 0)
        self.bigram_tag_frequencies = dict()
        self.emissionToCount = dict()
        self.corpus_length = 0

        for sentence in tagged_sentences:
            for wordTagPair in sentence:
                currWord = self.__update_word(wordTagPair[0], False)
                currTag = wordTagPair[1]

                self.corpus_length += 1
                self.word_frequencies[currWord] += 1
                self.tag_frequencies[currTag] += 1

            self.word_frequencies[Constant.SENTENCE_END] += 1
            self.tag_frequencies[Constant.SENTENCE_END_TAG] += 1

        if unk_threshold > 0:
            unk_unigram_word = set()
            keys = self.word_frequencies.keys()
            for key in keys:
                if self.word_frequencies[key] <= unk_threshold:
                    self.word_frequencies[Constant.UNK] += self.word_frequencies.get(key)
                    unk_unigram_word.add(key)

            for key in unk_unigram_word:
                del self.word_frequencies[key]

        for sentence in tagged_sentences:
            prevWord = Constant.SENTENCE_START
            prevTag = Constant.SENTENCE_START_TAG
            self.tag_frequencies[prevTag] += 1

            for wordTagPair in sentence:
                currWord = self.__update_word(wordTagPair[0])
                currTag = wordTagPair[1]

                if prevTag not in self.bigram_tag_frequencies:
                    self.bigram_tag_frequencies[prevTag] = defaultdict(lambda: 0)
                self.bigram_tag_frequencies[prevTag][currTag] += 1

                if currWord not in self.emissionToCount:
                    self.emissionToCount[currWord] = defaultdict(lambda: 0)
                self.emissionToCount[currWord][currTag] += 1

                prevWord = currWord
                prevTag = currTag

            currWord = Constant.SENTENCE_END
            currTag = Constant.SENTENCE_END_TAG

            if prevTag not in self.bigram_tag_frequencies:
                self.bigram_tag_frequencies[prevTag] = defaultdict(lambda: 0)
            self.bigram_tag_frequencies[prevTag][currTag] += 1

            if currWord not in self.emissionToCount:
                self.emissionToCount[currWord] = defaultdict(lambda: 0)
            self.emissionToCount[currWord][currTag] += 1

    def __update_word(self, word, handle_UNK = True):
        if word.startswith('\\u'):
            return 'Emotion'
        elif word.startswith('http'):
            return 'URL'
        elif word.startswith('@'):
            return 'AtPerson'
        elif word.startswith('#'):
            return 'HashTag'
        elif handle_UNK and word not in self.word_frequencies:
            return Constant.UNK
        else:
            return word

    def get_emission_prob(self, emission_data):
        word, tag = emission_data
        word = self.__update_word(word)

        return math.log(self.emissionToCount[word][tag] + self.k1) - math.log(self.tag_frequencies[tag] + (self.k1 * len(self.tag_frequencies)))

    def get_transition_prob(self, transition_data):
        return self.l1 * self.__get_transition_prob_unigram(transition_data) + \
               self.l2 * self.__get_transition_prob_bigram(transition_data)

    def __get_transition_prob_unigram(self, transition_data):
        prevTag, currTag = transition_data
        return math.log(self.tag_frequencies[currTag]) - math.log(self.corpus_length)

    def __get_transition_prob_bigram(self, transition_data):
        prevTag, currTag = transition_data

        if prevTag not in self.bigram_tag_frequencies:
            return float('-infinity')
        return math.log(self.bigram_tag_frequencies[prevTag][currTag] + self.k2) - math.log(sum(self.bigram_tag_frequencies[prevTag].values()) + (self.k2 * len(self.bigram_tag_frequencies.keys())))

    def predict(self, sentences):
        result = []
        for sentence in sentences:
            path = self.__viterbi(sentence)
            result.append(path)

        return result

    def __viterbi(self, sentence):
        wordCount = len(sentence)
        viterbiMatrix = []
        backpointerMatrix = []

        ########### Initialize #################
        for i in range(wordCount):
            viterbi = defaultdict(lambda: float("-infinity"))       # log(0) = float('-infinity')
            states = {}
            for tag in self.tag_frequencies.keys():
                if i == 0 and tag == Constant.SENTENCE_START_TAG:
                    viterbi[tag] = 0.0                              # log(1) = 0
                    states[tag] = Constant.SENTENCE_START_TAG
                else:
                    viterbi[tag] = float("-infinity")
                    states[tag] = 'placeholder'
            viterbiMatrix.append(viterbi)
            backpointerMatrix.append(states)

        ########### Recursion #################
        for t in range(1, wordCount):
            word = sentence[t]

            for currTag in self.tag_frequencies.keys():
                for prevTag in self.tag_frequencies.keys():
                    viterbi = viterbiMatrix[t - 1][prevTag] + self.get_transition_prob((prevTag, currTag)) + self.get_emission_prob((word, currTag))
                    if viterbi > viterbiMatrix[t][currTag]:
                        viterbiMatrix[t][currTag] = viterbi
                        backpointerMatrix[t][currTag] = prevTag

        ########### Termination #################
        bestTag = 'placeholder2'
        for tag in self.tag_frequencies.keys():
            if viterbiMatrix[wordCount - 1][tag] > viterbiMatrix[wordCount - 1][bestTag]:
                bestTag = tag

        path = [bestTag]
        for t in range(wordCount - 1, 1, -1):
            bestTag = backpointerMatrix[t][bestTag]
            path[0:0] = [bestTag]

        return path

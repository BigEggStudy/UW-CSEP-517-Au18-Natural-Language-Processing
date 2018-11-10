import math
import sys
from collections import defaultdict
import Constant
from joblib import Parallel, delayed

class TrigramHMM:
    def __init__(self, l1, l2, l3):
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3

    def fit(self, sentences, unk_threshold = 1):
        self.tag_frequencies = defaultdict(lambda: 0)
        self.word_frequencies = defaultdict(lambda: 0)
        self.bigram_tag_frequencies = dict()
        self.trigram_tag_frequencies = dict()
        self.emissionToCount = dict()
        self.corpus_length = 0

        for sentence in sentences:
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

        for sentence in sentences:
            prevTag1 = Constant.SENTENCE_START1_TAG
            prevTag2 = Constant.SENTENCE_START2_TAG
            self.tag_frequencies[prevTag1] += 1
            self.tag_frequencies[prevTag2] += 1

            if prevTag1 not in self.bigram_tag_frequencies:
                self.bigram_tag_frequencies[prevTag1] = defaultdict(lambda: 0)
            self.bigram_tag_frequencies[prevTag1][prevTag2] += 1

            for wordTagPair in sentence:
                currWord = self.__update_word(wordTagPair[0])
                currTag = wordTagPair[1]

                if (prevTag1, prevTag2) not in self.trigram_tag_frequencies:
                    self.trigram_tag_frequencies[(prevTag1, prevTag2)] = defaultdict(lambda: 0)
                self.trigram_tag_frequencies[(prevTag1, prevTag2)][currTag] += 1

                if prevTag2 not in self.bigram_tag_frequencies:
                    self.bigram_tag_frequencies[prevTag2] = defaultdict(lambda: 0)
                self.bigram_tag_frequencies[prevTag2][currTag] += 1

                if currWord not in self.emissionToCount:
                    self.emissionToCount[currWord] = defaultdict(lambda: 0)
                self.emissionToCount[currWord][currTag] += 1

                prevTag1 = prevTag2
                prevTag2 = currTag

            currWord = Constant.SENTENCE_END
            currTag = Constant.SENTENCE_END_TAG

            if (prevTag1, prevTag2) not in self.trigram_tag_frequencies:
                self.trigram_tag_frequencies[(prevTag1, prevTag2)] = defaultdict(lambda: 0)
            self.trigram_tag_frequencies[(prevTag1, prevTag2)][currTag] += 1

            if prevTag2 not in self.bigram_tag_frequencies:
                self.bigram_tag_frequencies[prevTag2] = defaultdict(lambda: 0)
            self.bigram_tag_frequencies[prevTag2][currTag] += 1

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

        if self.emissionToCount[word][tag] == 0:
            return float('-infinity')
        return math.log(self.emissionToCount[word][tag]) - math.log(self.tag_frequencies[tag])

    def get_transition_prob(self, transition_data):
        prob = self.l1 * self.__get_transition_prob_unigram(transition_data) + \
               self.l2 * self.__get_transition_prob_bigram(transition_data) + \
               self.l3 * self.__get_transition_prob_trigram(transition_data)
        return math.log(prob) if prob > 0 else float('-infinity')

    def __get_transition_prob_unigram(self, transition_data):
        prevTag1, prevTag2, currTag = transition_data

        return float(self.tag_frequencies[currTag]) / float(self.corpus_length)

    def __get_transition_prob_bigram(self, transition_data):
        prevTag1, prevTag2, currTag = transition_data

        if prevTag2 not in self.bigram_tag_frequencies or self.bigram_tag_frequencies[prevTag2][currTag] == 0:
            return 0.0
        return float(self.bigram_tag_frequencies[prevTag2][currTag]) / float(sum(self.bigram_tag_frequencies[prevTag2].values()))

    def __get_transition_prob_trigram(self, transition_data):
        prevTag1, prevTag2, currTag = transition_data

        if (prevTag1, prevTag2) not in self.trigram_tag_frequencies or self.trigram_tag_frequencies[(prevTag1, prevTag2)][currTag] == 0:
            return 0.0
        return float(self.trigram_tag_frequencies[(prevTag1, prevTag2)][currTag]) / float(sum(self.trigram_tag_frequencies[(prevTag1, prevTag2)].values()))

    def predict(self, sentences):
        result = []

        def __viterbi(sentence):
            wordCount = len(sentence)
            viterbiMatrix = []
            backpointerMatrix = []

            keys = self.tag_frequencies.keys()
            tags = [ (key1, key2) for key1 in keys for key2 in keys ]

            ########### Initialize #################
            for i in range(wordCount):
                viterbi = defaultdict(lambda: float('-infinity'))       # log(0) = float('-infinity')
                states = {}
                for tag1, tag2 in tags:
                    if i == 1 and tag1 == Constant.SENTENCE_START1_TAG and tag2 == Constant.SENTENCE_START2_TAG:
                        viterbi[(Constant.SENTENCE_START1_TAG, Constant.SENTENCE_START2_TAG)] = 0.0                              # log(1) = 0
                        states[Constant.SENTENCE_START2_TAG] = Constant.SENTENCE_START1_TAG
                    else:
                        viterbi[(tag1, tag2)] = float('-infinity')
                        states[tag2] = 'placeholder'
                viterbiMatrix.append(viterbi)
                backpointerMatrix.append(states)

            ########### Recursion #################
            def compute_viterbi(lastViterbi, prevTag1, prevTag2, currTag, word):
                value = lastViterbi + self.get_transition_prob((prevTag1, prevTag2, currTag)) + self.get_emission_prob((word, currTag))
                return (value, currTag)

            for t in range(2, wordCount):
                word = sentence[t]

                for (prevTag1, prevTag2) in tags:
                    if viterbiMatrix[t - 1][(prevTag1, prevTag2)] == float('-infinity'):
                        continue
                    result = [ compute_viterbi(viterbiMatrix[t - 1][(prevTag1, prevTag2)], prevTag1, prevTag2, currTag, word) for currTag in keys ]

                    (viterbi, currTag) = max(result, key=lambda p: p[0])
                    viterbiMatrix[t][(prevTag2, currTag)] = viterbi
                    backpointerMatrix[t][currTag] = prevTag2


            ########### Termination #################
            best_tag = max([(value, key) for key, value in viterbiMatrix[wordCount - 1].items()])[1][1]

            path = [ best_tag ]
            for t in range(wordCount - 1, 2, -1):
                best_tag = backpointerMatrix[t][best_tag]
                path[0:0] = [ best_tag ]

            return path

        result = Parallel(n_jobs=6)(delayed(__viterbi)(sentence) for sentence in sentences)
        return result

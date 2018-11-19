from collections import defaultdict, Counter
import random
import numpy as np
import re
from joblib import Parallel, delayed

import constant
from utils import cache

class StructuredPerceptron(object):
    def __init__(self, seed=100):
        """
        initialize model
        :return:
        """
        self.feature_weights = defaultdict(float)
        self.all_ner_tags = set()

        random.seed(seed)
        np.random.seed(seed)

    def fit(self, train_data, iterations=5, learning_rate=0.2):
        """
        read in a CoNLL file, extract emission features iterate over instances to train weight vector
        :param file_name:
        :return:
        """
        averaged_weights = Counter()

        for iteration in range(iterations):
            for i, (inputs, ner_tags) in enumerate(train_data):

                for ner_tag in ner_tags:
                    self.all_ner_tags.add(ner_tag)

                prediction = self.viterbi(inputs)

                # derive global features
                global_gold_features = self.__get_global_features(inputs, ner_tags)
                global_prediction_features = self.__get_global_features(inputs, prediction)

                # update weight vector
                for fid, count in global_gold_features.items():
                    self.feature_weights[fid] += learning_rate * count
                for fid, count in global_prediction_features.items():
                    self.feature_weights[fid] -= learning_rate * count

            averaged_weights.update(self.feature_weights)
            random.shuffle(train_data)

        self.feature_weights = averaged_weights

    def __get_global_features(self, inputs, ner_tags):
        feature_counts = Counter()

        for i, (input_data, ner_tag) in enumerate(zip(inputs, ner_tags)):
            previous_tag = constant.SENTENCE_START_TAG if i == 0 else ner_tags[i-1]
            feature_counts.update(self.get_features(input_data, ner_tag, previous_tag))

        return feature_counts

    def viterbi(self, sentence):
        N = len(sentence)
        M = len(self.all_ner_tags)
        tags = list(self.all_ner_tags)

        ########### Initialize #################
        viterbiMatrix = np.ones((M, N)) * float('-Inf')
        backpointerMatrix = np.ones((M, N), dtype=np.int16) * -1

        for j in range(M):
            cur_tag = tags[j]
            features = self.get_features(sentence[0], cur_tag, constant.SENTENCE_START_TAG)
            weights = sum((self.feature_weights[x] for x in features))
            viterbiMatrix[j, 0] = weights

        ########### Recursion #################
        for i in range(1, N):
            for j in range(M):
                tag = tags[j]
                best_score = float('-Inf')

                for k in range(M):
                    previous_tag = tags[k]

                    best_before = viterbiMatrix[k, i-1]

                    features = self.get_features(sentence[i], tag, previous_tag)
                    weights = sum((self.feature_weights[x] for x in features))
                    score = best_before + weights

                    if score > best_score:
                        viterbiMatrix[j, i] = score
                        best_score = score
                        backpointerMatrix[j, i] = k

        ########### Termination #################
        best_id = np.argmax(viterbiMatrix[:,-1])
        predtags = []
        predtags.append(tags[best_id])
        for i in range(N - 1, 0, -1):
            idx = int(backpointerMatrix[best_id, i])
            predtags.append(tags[idx])
            best_id = idx

        return predtags[::-1]

    def predict(self, test_data):
        result = Parallel(n_jobs=6)(delayed(self.viterbi)(sentence) for sentence in test_data)
        return result

    @cache
    def get_features(self, input_data, ner_tag, previous_ner_tag):
        word = input_data[0]
        pos_tag = input_data[1]
        syntactic_chunk_tag = input_data[2]
        word_lower = word.lower()
        prefix = word_lower[:3]
        suffix = word_lower[-3:]

        features = [
                    'TAG_%s' % (ner_tag),
                    'TAG_BIGRAM_%s_%s' % (previous_ner_tag, ner_tag),
                    'WORD+TAG_%s_%s' % (word, ner_tag),
                    'WORD_LOWER+TAG_%s_%s' % (word_lower, ner_tag),
                    'UPPER_%s_%s' % (word[0].isupper(), ner_tag),
                    'DASH_%s_%s' % ('-' in word, ner_tag),
                    'PREFIX+TAG_%s_%s' % (prefix, ner_tag),
                    'SUFFIX+TAG_%s_%s' % (suffix, ner_tag),
                    ('WORDSHAPE', self.shape(word), ner_tag),
                    'WORD+TAG_BIGRAM_%s_%s_%s' % (word, previous_ner_tag, ner_tag),
                    'SUFFIX+2TAGS_%s_%s_%s' % (suffix, previous_ner_tag, ner_tag),
                    'PREFIX+2TAGS_%s_%s_%s' % (prefix, previous_ner_tag, ner_tag),
                    'POSTAG+TAG_%s_%s' % (pos_tag, ner_tag),
                    'SYNTACTICTAG+TAG_%s_%s' % (syntactic_chunk_tag, ner_tag),
                    'POSTAG+SYNTACTICTAG+TAG_%s_%s_%s' % (pos_tag, syntactic_chunk_tag, ner_tag),
        ]
        return features

    @cache
    def shape(self, word):
        result = []
        for c in word:
            if c.isupper():
                result.append('X')
            elif c.islower():
                result.append('x')
            elif c in '0123456789':
                result.append('d')
            else:
                result.append(c)
        # replace multiple occurrences of a character with 'x*' and return it
        return re.sub(r"x+", "x*", ''.join(result))

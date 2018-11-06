import Constant

def read_sentences_from_file(file_path):
    result = []

    with open(file_path, 'r') as f:
        lines = f.readlines()

        for line in lines:
            result.append(line.split())

    return result

def prepare_data_unigram(sentences):
    for sentence in sentences:
        sentence.append(Constant.SENTENCE_END)
    return sentences

def prepare_data_bigram(sentences):
    for sentence in sentences:
        sentence.insert(0, Constant.SENTENCE_START)
        sentence.append(Constant.SENTENCE_END)
    return sentences

def prepare_data_trigram(sentences):
    for sentence in sentences:
        sentence.insert(0, Constant.SENTENCE_START1)
        sentence.insert(1, Constant.SENTENCE_START2)
        sentence.append(Constant.SENTENCE_END)
    return sentences

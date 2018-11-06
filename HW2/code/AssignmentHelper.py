import json

def read_sentences_from_file(file_path):
    result = []

    with open(file_path, 'r') as f:
        for line in f:
            result.append(json.loads(line))

    return result

def extract_labels(data):
    sentences = []
    labels = []

    for line in data:
        sentence = []
        label = []
        for (word, tag) in line:
            sentence.append(word)
            label.append(tag)
        sentences.append(sentence)
        labels.append(label)

    return (sentences, labels)

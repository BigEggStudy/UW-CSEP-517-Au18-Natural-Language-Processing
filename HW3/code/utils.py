def load_file(path):
    current_inputs = []
    current_ner_tags = []

    with open(path, 'r+') as f:
        for line in f:
            line = line.strip()
            if not line:
                if 'DOCSTART' not in current_inputs[0][0]:
                    yield (current_inputs, current_ner_tags)
                current_inputs = []
                current_ner_tags = []
            else:
                (word, pos_tag, syntactic_chunk_tag, ner_tag) = line.split()
                current_inputs.append(tuple([word, pos_tag, syntactic_chunk_tag]))
                current_ner_tags.append(ner_tag)
        if len(current_ner_tags) > 0 and 'DOCSTART' not in current_inputs[0][0]:
            yield (current_inputs, current_ner_tags)

def write_file(path, data, predict):
    with open(path,'w+') as f:
        for ((sentence_datas, ner_tags), predict_tags) in zip(data, predict):
            for ((word, pos_tag, syntactic_chunk_tag), ner_tag, predict_ner_tag) in zip(sentence_datas, ner_tags, predict_tags):
                f.write(f'{word} {pos_tag} {syntactic_chunk_tag} {ner_tag} {predict_ner_tag}\n')

def cache(f):
    cache_data = {}
    def helper(*args):
        key = tuple(args[1:])
        try:
            return cache_data[key]
        except KeyError:
            cache_data[key] = f(*args)
            return cache_data[key]
    return helper

def predict_eval(expected_value, predict_value, output = False):
    correct = 0
    total = 0

    for i, tags in enumerate(expected_value):
        if output:
            for (expected, predict) in zip(tags, predict_value[i]):
                print(f"{expected}\t{predict}")
        correct += sum([1 for (expected, predict) in zip(tags, predict_value[i]) if expected == predict])
        total += len(tags)
    return (correct, total)

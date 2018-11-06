from collections import defaultdict

def calculate_accuracy(expected, predicted):
    if len(expected) != len(predicted):
        print('ERROR: Expected and actual labels lengths dont match')
        return (0, 0, 0.0)

    totalcount = 0
    totalmatch = 0
    for i in range(len(expected)):
        for j in range(len(expected[i])):
            totalcount += 1
            if expected[i][j] == predicted[i][j]:
                totalmatch += 1

    return (totalmatch, totalcount, 100 * (totalmatch / totalcount))

def calculate_confusion_matrix(expected, predicted):
    if len(expected) != len(predicted):
        print('ERROR: Expected and actual labels lengths dont match')
        return None

    confusion_matrix = dict()
    expected_tags = set()
    predicted_tags = set()
    for i in range(len(expected)):
        for j in range(len(expected[i])):
            if expected[i][j] not in confusion_matrix:
                confusion_matrix[expected[i][j]] = defaultdict(lambda: 0)
            confusion_matrix[expected[i][j]][predicted[i][j]] += 1
            expected_tags.add(expected[i][j])
            predicted_tags.add(predicted[i][j])

    return (confusion_matrix, expected_tags, predicted_tags)

def get_most_error_data(expected, predicted, data):
    if len(expected) != len(predicted):
        print('ERROR: Expected and actual labels lengths dont match')
        return None

    sentences_id = []
    for i in range(len(expected)):
        error_count = 0
        for j in range(len(expected[i])):
            if expected[i][j] != predicted[i][j]:
                error_count += 1
        sentences_id.append((i, error_count))

    large_ten = sorted(sentences_id, key=lambda item: item[1], reverse=True)[:10]
    sentences = []
    for (id, error) in large_ten:
        sentences.append((data[id][1:-1], expected[id], predicted[id][:-1]))
    return sentences

def get_most_error_data_trigram(expected, predicted, data):
    if len(expected) != len(predicted):
        print('ERROR: Expected and actual labels lengths dont match')
        return None

    sentences_id = []
    for i in range(len(expected)):
        error_count = 0
        for j in range(len(expected[i])):
            if expected[i][j] != predicted[i][j]:
                error_count += 1
        sentences_id.append((i, error_count))

    large_ten = sorted(sentences_id, key=lambda item: item[1], reverse=True)[:10]
    sentences = []
    for (id, error) in large_ten:
        sentences.append((data[id][2:-1], expected[id], predicted[id][:-1]))
    return sentences

import itertools

import HMM
import algorithms_2


# Getting the sentences from the file specified in filename
def get_words(filename):
    file = open(filename, "rt")

    # Create the list that will contain the sentences in the file
    words = []
    for line in file:
        line = line.rstrip("\n")
        [word, tag] = line.split('/')  # Split every line into word and tag
        words.append((word, tag))  # Otherwise append in the last sentence the tuple made by word and tag

    return words


def split_data(words, separator):
    """
    Splits words into sentences according to the separator marker
    """
    sentences = []
    sentence = []
    for word, tag in words:
        if word == separator and tag == separator:
            if sentence:
                sentences.append(sentence)
            sentence = []
        else:
            sentence.append((word, tag))

    if sentence:
        sentences.append(sentence)

    return sentences


def evaluate_with_viterbi(transition_prob, output_prob, test_data, tags):
    """
    Prepares the data for the evaluation of the data according to the viterbi algorithm
    """
    correct = 0
    total = 0
    for sentence in test_data:
        untagged_sent = [x[0] for x in sentence]
        true_tags = [x[1] for x in sentence]
        path = algorithms_2.viterbi_alg(transition_prob, output_prob, untagged_sent, tags)
        pred_tags = [x[1] for x in path]
        #print("True:", true_tags)
        #print("Pred:", pred_tags)
        correct_tags = [p == t for p,t in zip(pred_tags, true_tags)]
        correct += sum(correct_tags)
        total += len(path)

    accuracy = correct/total
    print("The accuracy is:", accuracy)
    return accuracy


def get_unique_tags(data):
    """
    Returns the set of unique tags from the data that were provided in input
    """
    tags = []
    # Getting all the tags from the sentences
    for sentence in data:
            tags.append([x[1] for x in sentence])
    # Flattening the list
    tags = list(itertools.chain.from_iterable(tags))
    # Getting the unique tags from the list
    unique_tags = list(set(tags))

    return unique_tags


def set_split(split_num, splits, END_SENTENCE, words, split_sentences=True):
    """
    Splits the data into 3 sets: training, heldout and testing.
    How to split the data is defined by the array "split" which, contians the indices of the boundaries between
    each partition of the training data
    :param split_num: Current split index
    :param splits: array contatining splits
    :param END_SENTENCE: Separator that marks the end of each sentence
    :param words: raw data
    :param split_sentences: whether you should split sentences according to the END_SENTENCE marker or not
    :return:
    """
    print('-------   ', 'split', split_num, '   -------')

    data_to_split = [[], [], []]
    for j in range(3):
        data = []
        if j == 0:
            print('training: ', end=' ')
        elif j == 1:
            print('heldout: ', end=' ')
        elif j == 2:
            print('testing: ', end=' ')

        for k in range(len(splits[split_num][j])):
            print(splits[split_num][j][k][0], '-', splits[split_num][j][k][1], '+', end=' ')
            data.extend(words[splits[split_num][j][k][0]:splits[split_num][j][k][1]])
        print('')

        data_to_split[j].extend(data)

    if split_sentences:
        set_1 = split_data(data_to_split[0], END_SENTENCE)
        set_2 = split_data(data_to_split[1], END_SENTENCE)
        set_3 = split_data(data_to_split[2], END_SENTENCE)
    else:
        set_1 = data_to_split[0]
        set_2 = data_to_split[1]
        set_3 = data_to_split[2]

    return [set_1, set_2, set_3]


def main():
    filenames = ["TEXTCZ2.ptg"]
    END_SENTENCE = '###'
    type = 'bw'
    perc_held = 0.2

    FILENAME_OUT = "Results"+type+".txt"
    f_out = open(FILENAME_OUT, "w")

    for filename in filenames:
        print('##########    ', filename, '    ##########\n')
        f_out.write(filename + "\n")
        words = get_words(filename)

        if type == 'hmm':
            size_held = 20000
            size_test = 40000
            size_tr = len(words) - size_held - size_test
            splits = [[[[0, size_tr]], [[size_tr, size_held + size_tr]], [[size_held + size_tr, len(words)]]],
                      [[[size_test + size_held, len(words)]], [[size_test, size_test + size_held]], [[0, size_test]]],
                      [[[0, 20000], [20000 + size_test + size_held, len(words)]], [[20000, 20000 + size_held]],
                       [[20000 + size_held, 20000 + size_held + size_test]]],
                      [[[0, 60000], [60000 + size_test + size_held, len(words)]], [[60000, 60000 + size_held]],
                       [[60000 + size_held, 60000 + size_held + size_test]]],
                      [[[0, 100000], [100000 + size_test + size_held, len(words)]], [[100000, 100000 + size_held]],
                       [[100000 + size_held, 100000 + size_held + size_test]]]
                      ]

            for i in range(len(splits)):
                training_data, heldout_data, test_data = set_split(i, splits, END_SENTENCE, words)
                tags = get_unique_tags(training_data)
                [transition_prob, output_prob] = HMM.get_hmm_model(training_data, heldout_data)

                accuracy = evaluate_with_viterbi(transition_prob, output_prob, test_data, tags)
                f_out.write('split' + str(i) + ": " + str(accuracy) + "\n")

        else:
            size_held = 20000
            size_test = 40000
            size_tr = len(words) - size_held - size_test

            splits = [[[[0, size_tr]], [[size_tr, size_held + size_tr]], [[size_held + size_tr, len(words)]]],
                      [[[size_test + size_held, len(words)]], [[size_test, size_test + size_held]], [[0, size_test]]],
                      [[[0, 20000], [20000 + size_test + size_held, len(words)]], [[20000, 20000 + size_held]],
                       [[20000 + size_held, 20000 + size_held + size_test]]],
                      [[[0, 60000], [60000 + size_test + size_held, len(words)]], [[60000, 60000 + size_held]],
                       [[60000 + size_held, 60000 + size_held + size_test]]],
                      [[[0, 100000], [100000 + size_test + size_held, len(words)]], [[100000, 100000 + size_held]],
                       [[100000 + size_held, 100000 + size_held + size_test]]]
                      ]

            size_supervised = 500
            # We can change the size of the unsupervised training data by setting this variable.
            # This size was chosen accordingly to the power of my computer
            size_unsupervised = 1000

            for i in range(len(splits)):
                training_data, heldout_data, test_data = set_split(i, splits, END_SENTENCE, words, split_sentences=False)
                supervised_tr = [training_data[:size_supervised]]
                unsupervised_tr = [training_data[size_supervised: size_supervised + size_unsupervised]]
                heldout_data = [heldout_data]
                test_data = [test_data]

                tags = get_unique_tags(supervised_tr)
                [transition_prob, output_prob] = algorithms_2.baum_welch_alg(unsupervised_tr, supervised_tr,
                                                                             heldout_data, tags, perc_held)

                accuracy = evaluate_with_viterbi(transition_prob, output_prob, test_data, tags)
                f_out.write('split' + str(i) + ": " + str(accuracy) + "\n")
    return

main()
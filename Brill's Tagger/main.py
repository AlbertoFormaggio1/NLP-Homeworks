from itertools import chain
from nltk.tag import untag, brill, BrillTaggerTrainer, brill_trainer
from nltk.tag import DefaultTagger, UnigramTagger, TnT
import numpy as np

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


# Returns the tagger trained on the training data, the number of max rules is specified as input parameter
# You can specify also the initial tagger (can be either default, unigram, tnt)
# default_tag is the tag to use if 'Default' is chosen as initial tagger. NN is used since in several cases it
# is the most used one
def get_tagger(training_data, initial_tagger='default', default_tag='NN', smoothing_data=None):
    if not smoothing_data:
        smoothing_data = training_data

    # Setting the initial tags
    if initial_tagger == 'default':
        tag = DefaultTagger(default_tag)
    elif initial_tagger == 'unigram':
        tag = UnigramTagger(train=smoothing_data)
    elif initial_tagger == 'tnt':
        tag = TnT()
        tag.train(smoothing_data)
    else:
        raise ValueError('initial_tagger can be only either Default or Unigram')

    # Getting the templates from the nltk library
    #templates = brill.nltkdemo18()
    templates = brill.fntbl37()

    # Creating the TaggerTrainer by passing the initial parameters
    brill_tagger = brill_trainer.BrillTaggerTrainer(initial_tagger=tag, templates=templates)
    # Training the tagger on the training data
    trained_tagger = brill_tagger.train(training_data)
    return trained_tagger


def split_data(words, separator):
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


filenames = ["TEXTEN2.ptg", "TEXTCZ2.ptg"]
END_SENTENCE = '###'
taggers = ['default', 'unigram', 'tnt']

for filename in filenames:
    print('##########    ', filename, '    ##########\n')
    words = get_words(filename)

    size_held = 20000
    size_test = 40000
    size_tr = len(words) - size_held - size_test
    splits = [[[[0, size_tr]], [[size_tr, size_held + size_tr]], [[size_held + size_tr, len(words)]]],
              [[[size_test + size_held, len(words)]], [[size_test, size_test + size_held]], [[0, size_test]]],
              [[[0, 20000], [20000 + size_test + size_held, len(words)]], [[20000, 20000 + size_held]], [[20000 + size_held, 20000 + size_held + size_test]]],
              [[[0, 60000], [60000 + size_test + size_held, len(words)]], [[60000, 60000 + size_held]], [[60000 + size_held, 60000 + size_held + size_test]]],
              [[[0, 100000], [100000 + size_test + size_held, len(words)]], [[100000, 100000 + size_held]], [[100000 + size_held, 100000 + size_held + size_test]]]
              ]

    accuracy = {}
    for tagger in taggers:
        accuracy[tagger] = np.zeros(len(splits))

    for i in range(5):
        print('-------   ', 'split', i, '   -------')

        data_to_split = [[], [], []]
        for j in range(3):
            data = []
            if j == 0:
                print('training: ', end=' ')
            elif j == 1:
                print('heldout: ', end=' ')
            elif j == 2:
                print('testing: ', end=' ')

            for k in range(len(splits[i][j])):
                print(splits[i][j][k][0], '-', splits[i][j][k][1], '+', end=' ')
                data.extend(words[splits[i][j][k][0]:splits[i][j][k][1]])
            print('')

            data_to_split[j].extend(data)

        training_data = split_data(data_to_split[0], END_SENTENCE)
        heldout_data = split_data(data_to_split[1], END_SENTENCE)
        test_data = split_data(data_to_split[2], END_SENTENCE)

        for tagger in taggers:
            trained_tagger = get_tagger(training_data=training_data, initial_tagger=tagger, smoothing_data=heldout_data)
            accuracy[tagger][i] = trained_tagger.accuracy(test_data)
            print('Accuracy with *', tagger, '* baseline tagger:', accuracy[tagger][i])

    for tagger in taggers:
        print("Mean of accuracy with *", tagger, '* baseline tagger:', np.mean(accuracy[tagger]))
        print("Variance of accuracy with *", tagger, '* baseline tagger:', np.var(accuracy[tagger]))
    print(accuracy)

    print('\n\n')
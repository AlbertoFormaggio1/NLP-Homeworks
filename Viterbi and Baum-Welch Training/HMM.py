import numpy as np
import random


# Returns the HMM model (without smoothing). It consists of the transition probability and the output probability.
def get_hmm_model(training, heldout):
    """
    Returns the transition and output probabilities, smoothed, learned from the training set
    :param training: training set
    :param heldout: heldout set
    :return: smoothed transition probabilities, smoothed output probability
    """
    # Get counts from the data
    trigram_counts = get_trigram_counts(training)
    bigram_counts = get_bigram_counts(trigram_counts)
    unigram_counts = get_unigram_counts(bigram_counts)

    # get transition probabilities and respective lambdas
    trans_p = get_ngrams_from_counts(trigram_counts, bigram_counts, unigram_counts)
    lambdas_trans = em_algorithm(trans_p, heldout, type='transition')
    # Evaluate the smoothed transition probabilities
    smoothed_trans_p = get_smoothed_transition_probability(trans_p, lambdas_trans)

    # get the output probabilities
    out_p = get_output_probability(training)
    lambdas_out = em_algorithm(out_p, heldout, type='output')
    # Evaluate the smoothed output probabilities
    smoothed_output_p = get_smoothed_output_probability(out_p, lambdas_out)

    return [smoothed_trans_p, smoothed_output_p]


#region TRANSITION PROBABILITY COMPUTATION


def get_ngrams_from_counts(trigram_counts, bigram_counts, unigram_counts):
    """
    Get the n_grams probabilities from the counts for trigram, bigram and unigrams
    :param trigram_counts counts obtained from the trigram
    :param bigram_counts counts obtained from the bigram
    :param unigram_counts counts obtained from the unigram
    :return: [uniform_probability, unigram_probabilities, bigram_probabilities, trigram_probabilities]
    """
    # Copying the original trigram so I don't have to re-create the original structure
    trigram_prob = trigram_counts.copy()
    for k in trigram_prob.keys():
        for kk in trigram_counts[k]:
            for kkk in trigram_counts[k][kk]:
                # Compute the probability p(w_i | w_(i-2), w_(i-1)) = c(w_(i-2), w_(i-1), w_i) / c(w_(i-2), w_(i-1))
                trigram_prob[k][kk][kkk] = trigram_counts[k][kk][kkk] / bigram_counts[k][kk]

    # BIGRAM PROBABILITY
    bigram_prob = bigram_counts.copy()
    for k in bigram_prob.keys():
        for kk in bigram_counts[k].keys():
            bigram_prob[k][kk] = bigram_counts[k][kk] / unigram_counts[k]

    # UNIGRAM PROBABILITY
    unigram_prob = unigram_counts.copy()
    n_entries = sum(unigram_prob.values())  # length of the training data
    for k in unigram_prob.keys():
        unigram_prob[k] = unigram_counts[k] / n_entries

    # UNIFORM PROBABILITY
    # len(unigram_prob) is the size of the vocabulary
    unif_prob = 1 / len(unigram_prob)

    return [unif_prob, unigram_prob, bigram_prob, trigram_prob]


# Get the counts for the trigram directly from the data
def get_trigram_counts(sentences):
    counts = {}
    for sentence in sentences:
        # The first entry obtained from a sentence will be ('<START>', '<START>', firstTag)
        history = ["<START>", "<START>"]
        # Consider every entry in the sentence
        for word, tag in sentence:
            # Create the dictionary entry counts[history[0]][history[1]][tag], if not present already and set its value to 0
            if not counts.get(tag):
                counts[tag] = {}
            if not counts[tag].get(history[1]):
                counts[tag][history[1]] = {}
            if not counts[tag][history[1]].get(history[0]):
                counts[tag][history[1]][history[0]] = 0

            # Increment the counts
            counts[tag][history[1]][history[0]] += 1
            # Update the history
            history[0] = history[1]
            history[1] = tag

    return counts


def get_bigram_counts(trigram_counts):
    # tr[k][kk][kkk] ==> bi[k][kk]
    bigram_counts = {}
    for k in trigram_counts.keys():

        # Evaluating the bigram by summing over all the w_(i-2) over the counts of the trigram
        # c2(w_(i-1),w_(i)) = sum x over c3(x, w_(i-1), w_i).
        # where ci(x,y,z) is the count of the occurences of the sequence of tokens x,y,z.
        bigram_counts[k] = {}
        for kk in trigram_counts[k].keys():
            bigram_counts[k][kk] = sum(trigram_counts[k][kk].values())

    return bigram_counts


def get_unigram_counts(bigram_counts):
    unigram_counts = {}
    for k in bigram_counts.keys():
        unigram_counts[k] = sum(bigram_counts[k].values())

    return unigram_counts


def get_train_trans_prob(previous, current, p):
    """
    Returns the probabilities obtained from the training set or the uniform probability if the history is not present
    :param previous history 2 tags long
    :param current current tag to consider
    :param p vectors of probabilities. p = [uniform_prob, unigram_prob, bigram_prob, trigram_prob]
    """

    train_prob = np.zeros(4)

    # Get, if present, the trigram probability of the sequence  [previous[0], previous[1], current]
    train_prob[3] = p[3].get(current, {}).get(previous[1], {}).get(previous[0], 0)
    """ Point 3 in slide 93 of npfl067: 
    If p3(t,h) = 0, then assign uniform probability if c2(h) = 0 (hence, the history is not found in the bigram 
    distribution), otherwise leave the proability as it is"""
    if train_prob[3] == 0 and p[2].get(current, {}).get(previous[1], 0) == 0:
        train_prob[3] = p[0]

    # Get, if present, the trigram probability of the sequence  [previous[1], current]
    train_prob[2] = p[2].get(current, {}).get(previous[1], 0)
    """ Point 3 in slide 93 of npfl067: 
        If p2(t,h) = 0, then assign uniform probability if c1(h) = 0 (hence, the history is not found in the unigram 
        distribution), otherwise leave the proability as it is"""
    if train_prob[2] == 0 and p[1].get(current, 0) == 0:
        train_prob[2] = p[0]

    # Get the probability if the word is present, uniform_probability otherwise
    train_prob[1] = p[1].get(current, p[0])
    train_prob[0] = p[0]

    return train_prob


def get_train_out_prob(state, word, p):
    """
    Gets the probability from the training data of having word in output when in the state specified as input.
    If the probability is 0 because some sequence was never seen in the training data, the output will be the
    uniform probability
    :param state: Current state defined by [prevState, state]
    :param word: Word currently analyzed
    :param p: array [uniform, p(w), p(w|s)], where p(w) and p(w|s) are dictionaries
    :return: [uniform, p(word) or uniform, p(word|state) or uniform]
    """
    train_prob = np.zeros(3)
    # Get, if present, the probability of having word as output when in state s. Return 0 otherwise
    train_prob[2] = p[2].get(state[0], {}).get(state[1], {}).get(word, 0)
    """ Point 3 in slide 93 of npfl067: 
        If p2(t,h) = 0, then assign uniform probability if c1(h) = 0 (hence, the state is not found in the original 
        distribution), otherwise leave the proability as it is"""
    if train_prob[2] == 0 and p[1].get(word, 0) == 0:
        train_prob[2] = p[0]

    train_prob[1] = p[1].get(word, p[0])
    train_prob[0] = p[0]

    return train_prob


def get_smoothed_transition_probability(trans_p, lambdas):
    """
    Computes the transition probability p(t_i | t_(i-2), t_(i-1)) smoothed, thus computed by using the lambdas
    parameters obtained thanks to the EM algorithm
    :param trans_p: original transition probabilities for uniform, unigram, bigram, trigram
    :param lambdas: lambdas computed with EM
    :return: smoothed probability
    """

    for k in trans_p[3].keys():
        for kk in trans_p[3][k].keys():
            for kkk in trans_p[3][k][kk].keys():
                # evaluate the smoothed probability for the current history
                trans_p[3][k][kk][kkk] = lambdas[3] * trans_p[3][k][kk][kkk]
            trans_p[2][k][kk] = lambdas[2] * trans_p[2][k][kk]
        trans_p[1][k] = lambdas[1] * trans_p[1][k]

    trans_p[0] = lambdas[0] * trans_p[0]

    return trans_p

# endregion

# region OUTPUT PROBABILITY COMPUTATION

def get_output_probability(sentences):
    """
    Returns the dictionary with the output probabilities: p(w|s). Probability of having word w in state s.
    Since the history has length 2, our state will be a pair (prevTag, currentTag).
    :param sentences: sentences in the training set
    :return: the output probabilities
    """

    # Computing the probability of having the output probability p(w|s), hence the probability of having word w in
    # output in state s.
    counts_classes = get_classes_counts(sentences)
    p_classes = counts_classes.copy()
    for prev in p_classes.keys():
        for tag in p_classes[prev].keys():
            counts_state = sum(p_classes[prev][tag].values())
            for word in p_classes[prev][tag].keys():
                p_classes[prev][tag][word] = counts_classes[prev][tag][word] / counts_state

    # Computing the probability of having word w in the training data
    counts_words = get_words_counts(sentences)
    p_words = counts_words.copy()
    total_words = sum(counts_words.values())
    for w in counts_words:
        p_words[w] = counts_words[w] / total_words

    # Computing uniform probability
    vocab_len = len(p_words.keys())
    unif_prob = 1 / vocab_len

    return [unif_prob, p_words, p_classes]


def get_classes_counts(sentences):
    """
    Computes, for each word w, how many times it is returned in output given the current and the previous tag
    :param senences the sentences over which you need to compute the counts
    """
    classes_counts = {}
    for sentence in sentences:
        prev = '<START>'
        for word, tag in sentence:
            if not classes_counts.get(prev):
                classes_counts[prev] = {}
            if not classes_counts[prev].get(tag):
                classes_counts[prev][tag] = {}
            if not classes_counts[prev][tag].get(word):
                classes_counts[prev][tag][word] = 0

            classes_counts[prev][tag][word] += 1
            prev = tag

    return classes_counts


def get_words_counts(sentences):
    """
    Counts the number of words inside the input file. This is useful when computing the smoothed output probabilty
    with the em algorithm
    :param sentences: set of sentences over which you need to compute the counts of words
    :return: a dictionary where dict[w] are the counts associated to word w
    """
    words_counts = {}
    for sentence in sentences:
        for word, tag in sentence:
            if not words_counts.get(word):
                words_counts[word] = 0
            words_counts[word] += 1

    return words_counts

def get_smoothed_output_probability(out_p, lambdas):
    """
    Computes the smoothed output probability computed as follows: p = l2 * p(w|s) + l1 * p(w) + l0/|V|
    where:
     - p(w|s) is the probability of generating word w in state s
     - p(w) is the probability of generating word w
     - |V| is the size of the vocabulary, thus the number of unique words in the file
    :param out_p: output probability [uniform, p(w), p(w|s)]
    :param lambdas: lambdas computed by the smoothing EM_algorithm
    :return: the smoothed probability
    """

    for prevTag in out_p[2].keys():
        for tag in out_p[2][prevTag].keys():
            for word in out_p[2][prevTag][tag].keys():
                out_p[2][prevTag][tag][word] = lambdas[2] * out_p[2][prevTag][tag][word]

    for word in out_p[1].keys():
        out_p[1][word] = out_p[1][word] * lambdas[1]

    out_p[0] = out_p[0] * lambdas[0]

    return out_p


# endregion

# region EM_ALGORITHM

def em_algorithm(p, heldout, type, epsilon=0.001):
    """Returns the lambdas obtained from the EM algorithm by applying MLE over the heldout data

    :param p: the set of probabilities we want to smooth:
    if type is 'transition' => p = [uniform, unigram, bigram, trigram]
    if type is 'output' => p = [uniform, p(w), p(w|s)]
    :param heldout: data on which you need to set the parameters.
    :param epsilon: defines when the convergence is reached.
    :param type: 'transition' or 'output'. Based on this value, the probability will be treated accordingly
    """

    if type == 'transition':
        lambdas_len = 4
    else:
        lambdas_len = 3

    # Generating starting random lambdas
    params = []
    for i in range(0, lambdas_len):
        params.append(random.random())
    tot = sum(params)
    lambdas = [a / tot for a in params]

    # Do until convergence is reached (it is always reached: known from a proof related to Jensen's inequality)
    while 1:
        exp_counts = np.zeros(lambdas_len, dtype=float)

        for sentence in heldout:
            prev_tag = ["<START>", "<START>"]
            for word, tag in sentence:
                if type == 'transition':
                    # Computing probability over the training data. This is done only once here since the probabilities
                    # don't change along the iterations. They stay always the same.
                    train_prob = get_train_trans_prob(prev_tag, tag, p)
                else:
                    state = [prev_tag[1], tag]
                    train_prob = get_train_out_prob(state, word, p)
                # Evaluating probability p' = l3*p3 + l2*p2 + l1*p1 + l0*p0
                smoothed_prob = np.sum(lambdas * train_prob)

                # Updating value of the expected counts
                exp_counts += lambdas * train_prob / smoothed_prob

                # Updating the current history
                prev_tag[0] = prev_tag[1]
                prev_tag[1] = tag

        # Evaluate the new set of lambdas
        arr_exp_counts = np.array(exp_counts)
        new_l = arr_exp_counts / np.sum(arr_exp_counts)

        """See if convergence condition is reached: 
        Stop if |newl_i - l_i| < epsilon for each i=0...3"""
        convergence = True
        for i in range(len(new_l)):
            if abs(new_l[i] - lambdas[i]) >= epsilon:
                convergence = False

        # Update the set of lambdas
        lambdas = new_l

        # Convergence reached
        if convergence:
            break

    # Once convergence is reached, return the set of lambdas found
    print("The lambdas found for the", type, "probability are:", lambdas)
    print("The sum of the lambdas is", sum(lambdas))
    return lambdas

# endregion

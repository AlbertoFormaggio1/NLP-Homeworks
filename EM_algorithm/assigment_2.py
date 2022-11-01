import math
import random
def evaluate_trigram(training_data):
    trigram = {}
    # using empty space as initial values for handling the beginning of the file
    prevValues = ["", ""]
    # for each value inside the data
    for x in training_data:

        # Creating a dictionary for the current 3-tuple, if not yet present in the dictionary
        if not trigram.get(prevValues[0]):
            trigram[prevValues[0]] = {}
        if not trigram[prevValues[0]].get(prevValues[1]):
            trigram[prevValues[0]][prevValues[1]] = {}
        if not trigram[prevValues[0]][prevValues[1]].get(x):
            trigram[prevValues[0]][prevValues[1]][x] = 0

        # Adding 1 to the number of the occurences (prev, x), in this order
        trigram[prevValues[0]][prevValues[1]][x] += 1
        prevValues[0] = prevValues[1]
        prevValues[1] = x

    return trigram


def evaluate_bigram(trigram):
    bigram = {}
    for k in trigram.keys():
        # Reject the pair '','' as it is not in the original file, it was added to the trigram in order to handle
        # correctly the beginning of the string
        if k == '':
            continue

        # Evaluating the bigram by summing over all the w_(i-2) over the counts of the trigram
        # c2(w_(i-1),w_i) = sum x over c3(x, w_(i-1), w_i).
        # where ci(x,y,z) is the count of the occurences of the sequence of tokens x,y,z.
        bigram[k] = {}
        for kk in trigram[k].keys():
            bigram[k][kk] = sum(trigram[k][kk].values())
    return bigram


def evaluate_unigram(list):
    unigram = {}
    # the unigram can be easily computed from the data
    for el in list:
        unigram[el] = unigram.get(el, 0) + 1

    return unigram


def compute_trigram_probability(trigram, bigram):
    prob = trigram.copy()
    for k in prob.keys():
        # Rejecting, as usual, '' as first key as there is no entry in the bigram such that ('','')
        if k == '':
            continue

        for kk in prob[k].keys():
            for kkk in prob[k][kk].keys():
                # Compute the probability
                prob[k][kk][kkk] = trigram[k][kk][kkk] / bigram[k][kk]

    return prob


def compute_bigram_probability(bigram, unigram):
    prob = bigram.copy()
    for k in prob.keys():
        for kk in prob[k].keys():
            prob[k][kk] = bigram[k][kk] / unigram[k]

    return prob


def compute_unigram_probability(unigram, text_size):
    prob = unigram.copy()
    for k in prob.keys():
        prob[k] = unigram[k] / text_size

    return prob


def compute_parameters(tri_prob, bi_prob, uni_prob, unif_prob, test):
    # Generating starting random parameters
    params = []
    for i in range(0, 4):
        params.append(random.random())
    tot = sum(params)
    params = [a/tot for a in params]

    probs = {}
    # Value used for the termination condition |newlambda - lambda| < epsilon (for each lambda)
    EPSILON = 0.0001

    while 1:
        expected_counts = [0, 0, 0, 0]
        next_params = [0, 0, 0, 0]
        previousWords = ["", ""]

        #Evaluate the proabilities over the heldout data
        for word in test:
            if not probs.get(previousWords[0]):
                probs[previousWords[0]] = {}
            if not probs[previousWords[0]].get(previousWords[1]):
                probs[previousWords[0]][previousWords[1]] = {}

            train_prob = [0, 0, 0, 0]
            # Get, if present, the trigram probability of the current sequence of tokens
            train_prob[3] = tri_prob.get(previousWords[0], {}).get(previousWords[1], {}).get(word, 0)
            # If the sequence is not present but the bigram probability for the couple (wi-1, wi) is present, then leave 0.
            # Otherwise, assign a uniform probability since the pair has never been seen on the training set.
            if train_prob[3] == 0 and bi_prob.get(previousWords[0], {}).get(previousWords[1], 0) == 0:
                train_prob[3] = unif_prob

            # Same as before, but we have bigram and unigram instead of trigram and bigram
            train_prob[2] = bi_prob.get(previousWords[1], {}).get(word, 0)
            if train_prob[2] == 0 and uni_prob.get(previousWords[1], 0) == 0:
                train_prob[2] = unif_prob

            # Get the probability if the word is present, 0 otherwise
            train_prob[1] = uni_prob.get(word, 0)
            train_prob[0] = unif_prob

            # Evaluate p' using the lambda parameters
            tmp_prob = sum(params[i] * train_prob[i] for i in range(0, 4))
            probs[previousWords[0]][previousWords[1]][word] = tmp_prob

            # Modify expected counts by adding current corresponding value
            for i in range(0, 4):
                expected_counts[i] += params[i] * train_prob[i] / tmp_prob

            # Replace history with the current one
            previousWords[0] = previousWords[1]
            previousWords[1] = word

        # Evaluate next params with the known formula
        exp_sum = sum(expected_counts)
        for i in range(0, 4):
            next_params[i] = expected_counts[i] / exp_sum

        # If all the differences between new and old lambdas are lower than EPSILON, then I have reached convergence
        convergence = True
        for i in range(0, 4):
            if abs(params[i] - next_params[i]) > EPSILON:
                convergence = False

        params = next_params.copy()

        # If all the differences between lambdas are lower than EPSILON, then I have convergence and I can terminate the algorithm
        if convergence:
            break

    return params


def compute_smoothed_model_probability(tri_prob, bi_prob, uni_prob, unif_prob, test, params):
    previousWords = ["", ""]
    probs = {}

    for word in test:
        if not probs.get(previousWords[0]):
            probs[previousWords[0]] = {}
        if not probs[previousWords[0]].get(previousWords[1]):
            probs[previousWords[0]][previousWords[1]] = {}

        train_prob = [0, 0, 0, 0]
        train_prob[3] = tri_prob.get(previousWords[0], {}).get(previousWords[1], {}).get(word, 0)
        if train_prob[3] == 0 and bi_prob.get(previousWords[0], {}).get(previousWords[1], 0) == 0:
            train_prob[3] = unif_prob

        train_prob[2] = bi_prob.get(previousWords[1], {}).get(word, 0)
        if train_prob[2] == 0 and uni_prob.get(previousWords[1], 0) == 0:
            train_prob[2] = unif_prob

        train_prob[1] = uni_prob.get(word, 0)
        train_prob[0] = unif_prob

        # Evaluate p'
        tmp_prob = sum(params[i] * train_prob[i] for i in range(0, 4))

        probs[previousWords[0]][previousWords[1]][word] = tmp_prob

        previousWords[0] = previousWords[1]
        previousWords[1] = word

    return probs


def compute_cross_entropy(pred_distr, true_distr):
    entropy = 0

    for k in pred_distr.keys():
        for kk in pred_distr[k].keys():
            for kkk in pred_distr[k][kk].keys():
                entropy -= true_distr[k][kk][kkk] * math.log2(pred_distr[k][kk][kkk])

    return entropy


def main():
    FILENAME = "TEXTCZ1.txt"
    FILENAME_OUT = "Results.txt"

    f_out = open(FILENAME_OUT,"w")
    f = open(FILENAME, "r")

    #Splitting data
    initial_list = []
    count_words = 0
    test_size = 20000
    heldout_size = 40000

    for x in f:
        x = x[:-1]
        initial_list.append(x)
        count_words += 1

    training_data = initial_list[:-(test_size + heldout_size)]
    training_size = len(training_data)
    vocab_size = len(set(training_data))
    heldout_data = initial_list[-(test_size + heldout_size):-test_size]
    test_data = initial_list[-test_size:]

    # Compute the trigram and bigrams from the training data
    trigram = evaluate_trigram(training_data)
    bigram = evaluate_bigram(trigram)
    unigram = evaluate_unigram(training_data)

    # Now evaluate the probabilities for each of the n-gram considered
    tri_prob = compute_trigram_probability(trigram, bigram)
    bi_prob = compute_bigram_probability(bigram, unigram)
    uni_prob = compute_unigram_probability(unigram, training_size)
    unif_prob = 1 / vocab_size

    lambdas = compute_parameters(tri_prob, bi_prob, uni_prob, unif_prob, heldout_data)

    test_trigram = evaluate_trigram(test_data)
    test_bigram = evaluate_bigram(test_trigram)

    test_true_prob = compute_trigram_probability(test_trigram, test_bigram)

    test_model_prob = compute_smoothed_model_probability(tri_prob, bi_prob, uni_prob, unif_prob, test_data, lambdas)

    print("Computing the cross entropy with original parameters...")
    for i in range(0, 4):
        print('lambda'+str(i)+':', lambdas[i])
    cross_e = compute_cross_entropy(test_model_prob, test_true_prob)
    print(cross_e, "\n")
    f_out.write(str(cross_e) + "\n")

    # Increasing the value of lambda3
    perc = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    diff = 1 - lambdas[3]
    for p in perc:
        # Changing the value of lambda
        new_lambdas = lambdas.copy()
        to_add = p*diff
        new_lambdas[3] = lambdas[3] + to_add

        # Modify, uniformly all the other parameters
        to_remove = to_add/3
        while 1:
            # Remove the quantity to_remove if the parameter is not already at its minimum
            for i in range(0, 3):
                if new_lambdas[i] > 1e-8:
                    new_lambdas[i] -= to_remove

            # If all the parameters are positive, then I have convergence
            convergence = True
            for i in range(0, 3):
                if new_lambdas[i] <= 0:
                    convergence = False
                    break

            if convergence:
                break

            # If some parameters are negative, then count how many variables are still available for subtracting
            # the quantity to remove.
            # Get how much the data are negative and split this value into the remaining vars.
            # Using 1e-8 instead of 0 as value because when taking the logarithm, using 0 lead to some problems.
            remaining_vars = 0
            err = 0
            for i in range(0, 3):
                if new_lambdas[i] < 1e-8:
                    err += abs(new_lambdas[i]) + 1e-8
                    new_lambdas[i] = 1e-8
                elif new_lambdas[i] > 1e-8:
                    remaining_vars += 1

            to_remove = err/remaining_vars


        print("Computing the cross entropy after adding", str(p*100)+"%", "to trigram smoothing parameter...")
        print("The lambdas used are:")
        for i in range(0, 4):
            print("lambda"+str(i)+":", new_lambdas[i])
        test_model_prob = compute_smoothed_model_probability(tri_prob, bi_prob, uni_prob, unif_prob, test_data, new_lambdas)
        cross_e = compute_cross_entropy(test_model_prob, test_true_prob)
        print("The cross entropy is:", cross_e, "\n")
        f_out.write(str(cross_e) + "\n")

    # Decreasing the value of lambda3. The reasoning is the same as before, but with 1 as cap value instead of 0.
    perc = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]
    for p in perc:
        new_lambdas = lambdas.copy()
        to_remove = lambdas[3] * (1-p)
        new_lambdas[3] = lambdas[3] - to_remove

        to_add = to_remove / 3
        while 1:
            for i in range(0, 3):
                if new_lambdas[i] < 1:
                    new_lambdas[i] += to_add

            convergence = True
            for i in range(0, 3):
                if new_lambdas[i] > 1:
                    convergence = False
                    break

            if convergence:
                break

            remaining_vars = 0
            err = 0
            for i in range(0, 3):
                if new_lambdas[i] > 1:
                    err += new_lambdas[i] - 1
                    new_lambdas[i] = 1
                elif new_lambdas[i] < 1:
                    remaining_vars += 1

            to_add = err / remaining_vars


        print("Computing the cross entropy after removing", str((1-p) * 100) + "%", "to trigram smoothing parameter...")
        for i in range(0, 4):
            print("lambda" + str(i) + ":", new_lambdas[i])
        test_model_prob = compute_smoothed_model_probability(tri_prob, bi_prob, uni_prob, unif_prob, test_data,
                                                             new_lambdas)
        cross_e = compute_cross_entropy(test_model_prob, test_true_prob)
        print("The cross entropy is:", cross_e, "\n")
        f_out.write(str(cross_e) + "\n")




main()

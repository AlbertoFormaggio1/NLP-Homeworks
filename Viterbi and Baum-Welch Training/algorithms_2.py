import math
import HMM
import numpy as np


def viterbi_alg(trans_p, out_p, sentence, tags):
    vit = [{}]
    initial_alpha = 1
    # Initialize initial state with alpha=1
    vit[0]["<START>", "<START>"] = {"alpha": initial_alpha, "prev": None}

    # For every observation done in the data, you will have one time instant "t".
    # Forward pass
    for t in range(1, len(sentence) + 1):
        vit.append({})
        # Obtaining the states "valid" in the previous time instant
        prev_keys = list(vit[t - 1])
        # For every current state that I can have (defined by the pair (t_i, t_i-1) )
        valid_prev_keys = [x[1] for x in prev_keys]
        for prev_t in valid_prev_keys:
            for cur_t in tags:
                # Computing the emission probability. This doesn't depend on the previous state, so we're computing
                # it once at the beginning of each iteration
                emm_prob = out_p[2].get(prev_t, {}).get(cur_t, {}).get(sentence[t-1], 0) \
                           + out_p[1].get(sentence[t-1], 0) \
                           + out_p[0]

                # prev_keys is a list of pairs are [t_(i-2), t_(i-1)]
                # remember that trans_p is such that p(t_(i-2), t_(i-1), t_i) = trans[t_i][t_(i-1)][t_(i-2)]
                # We are now initializing the max_alpha as the first alpha in the list
                cur_trans_prob = trans_p[3].get(cur_t, {}).get(prev_t, {}).get(prev_keys[0][0], 0) \
                                  + trans_p[2].get(cur_t, {}).get(prev_t, 0) \
                                  + trans_p[1].get(cur_t, 0) \
                                  + trans_p[0]

                # In order to avoid underflow errors, we compute the logarithm (log is a monotonic function, which doesn't
                # change the result of the multiplication
                emm_prob = np.log(emm_prob)
                cur_trans_prob = np.log(cur_trans_prob)

                # we sum instead of multiplying since we are considering the logarithm (after applying the property of
                # the logarithms to the original viterbi formula we get this formula below
                max_alpha = vit[t-1][prev_keys[0]]["alpha"] + cur_trans_prob + emm_prob
                # Set the best previous state as the first state in the previous layer
                best_prev_state = prev_keys[0]

                # For every valid previous state, evaluate the alpha
                for st in prev_keys[1:]:
                    # I am looking for the previous states where the 2nd tag is equal to prev.
                    # So I want st = (st[0], st[1]) = (st[0], prev). If that's not the case, this node will be considered
                    # another time. We have the constraint that st[1] = prev!!!
                    if st[1] != prev_t:
                        continue

                    # Evaluating the current alpha
                    cur_trans_prob = trans_p[3].get(cur_t, {}).get(prev_t, {}).get(st[0], 0) \
                                     + trans_p[2].get(cur_t, {}).get(prev_t, 0) \
                                     + trans_p[1].get(cur_t, 0) \
                                     + trans_p[0]


                    cur_trans_prob = np.log(cur_trans_prob)
                    alpha = vit[t-1][st]["alpha"] + cur_trans_prob + emm_prob

                    # If the alpha just computed is the maximum one up until now, update the alpha and the best previous state
                    if alpha > max_alpha:
                        max_alpha = alpha
                        best_prev_state = st

                # Current state is (prev_state[1],cur)
                vit[t][prev_t, cur_t] = {"alpha": max_alpha, "prev": best_prev_state}
        vit[t] = pruning(vit[t])

    # Backward pass
    # First, we need to evaluate the maximum value in the last step
    max_prob = float('-inf')
    best_state = None
    for st, dict in vit[-1].items():
        if dict["alpha"] > max_prob:
            max_prob = dict["alpha"]
            best_state = st

    path = []
    path.append(best_state)
    prev = best_state

    # Appending at the end of the list and then reverse the sequence for better performance
    # (Remember that best_state is the state in the rightmost part of the Trellis, so you have to go backward)
    for i in range(len(vit) - 2, 0, -1):
        path.append(vit[i+1][prev]["prev"])
        prev = vit[i+1][prev]["prev"]

    path.reverse()

    #print("The path is", path, "obtained with probability", max_prob)

    return path


def pruning(current_viterbi, threshold=10):
    """
    Removes the nodes in the viterbi graph that have an alpha which is not in the top-10 of the highest alphas
    :param current_viterbi: nodes at the current step of the viterbi graph
    :param threshold: number of nodes to keep
    :return:
    """
    # Sorting the elements by alpha
    sorted_viterbi = sorted(current_viterbi.items(), key=lambda x: x[1]["alpha"])
    pruned_viterbi = current_viterbi.copy()

    # If the current level has more than <threshold> states, remove the states associated to the smallest alphas
    for item in sorted_viterbi:
        if len(pruned_viterbi) < threshold:
            break

        pruned_viterbi.pop(item[0])

    return pruned_viterbi


def untag(sentences):
    """
    Remove the tags from the training data so they are ready for the unuspervised learning phase
    """
    untagged_sentences = []
    for sentence in sentences:
        untagged_sentence = [x[0] for x in sentence]
        untagged_sentences.append(untagged_sentence)

    return untagged_sentences


def baum_welch_alg(unsupervised_training, sup_training, heldout, tags, perc_held=0.05, epsilon=0.4):
    """
    Estimates the updates for the probabilities by means of the Baum-Welch algorithm.
    The formulae used can be found at https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm
    :param unsupervised_training: Data to use for the unsupervised training
    :param sup_training: Data to use for the initial estimation of the HMM parameters
    :param heldout: Heldout data to use at the end for smoothing
    :param tags: List of unique tags
    :param perc_held: Percentage of heldout data to obtain from the supervised learning
    :param epsilon: Threshold for convergence
    :return: Smoothed transition and output probabilties
    """
    # Initialization of trans_p, out_p:
    # - all transitions are allowed with uniform probability
    # - we need to set the output probability based on the training data
    num_held = int(len(sup_training[0]) * perc_held)
    heldout_tr = [sup_training[0][:num_held]]
    actual_training = [sup_training[0][num_held:]]
    unsupervised_training = untag(unsupervised_training)

    [strans_p, sout_p] = HMM.get_hmm_model(actual_training, heldout_tr)

    convergence = 1000
    while True:
        for sentence in unsupervised_training:
            sent = sentence.copy()
            # Inserting an element in position 0 in such a way that the whole list is traslated of 1 unit to the right.
            # This way the formulae hold by using index t=1...N
            sent.insert(0, 'NAN')
            # Computing forward probabilities
            alpha, normalization_fact = compute_alphas(sent, tags, strans_p, sout_p)
            # Computing backward probabilities
            beta = compute_betas(sent, tags, strans_p, sout_p, normalization_fact)

            [csi, gamma_out, gamma_state] = compute_csi_gamma(sent, strans_p, alpha, beta, sout_p)

            convergence = 1000

            # Obtaining the set of unique states from csi
            set_keys_tr = set()
            for t in range(len(sent) - 1):
                for k in csi[t].keys():
                    set_keys_tr.add(k)

            # Updating the transition probabilities. Computing the value aij* reported on the webpage
            for prev_t, cur_t, next_t in set_keys_tr:
                sum_csi = 0
                sum_gamma = 0
                for t in range(len(sent) - 1):
                    sum_csi += csi[t].get((prev_t, cur_t, next_t), 0)
                    sum_gamma += gamma_state[t].get((prev_t, cur_t), 0)
                # This shouldn't be 0 anyway
                if sum_csi == 0 or sum_gamma == 0:
                    continue
                new_trans_p = sum_csi / sum_gamma

                if not strans_p[3].get(next_t):
                    strans_p[3][next_t] = {}
                if not strans_p[3][next_t].get(cur_t):
                    strans_p[3][next_t][cur_t] = {}
                if not strans_p[3][next_t][cur_t].get(prev_t):
                    strans_p[3][next_t][next_t][prev_t] = 0

                # If the update is higher than epsilon, decrease the number of errors that can be still made
                if abs(new_trans_p - strans_p[3][next_t][cur_t][prev_t]) > epsilon:
                    convergence -= 1

                strans_p[3][next_t][cur_t][prev_t] = new_trans_p


            # Computing the formula for the updates of the output probabilities
            set_keys_obs = set()
            for t in range(len(sent)):
                for k in gamma_out[t].keys():
                    set_keys_obs.add(k)

            for word, cur_t, next_t in set_keys_obs:
                sum_gamma_obs = 0
                sum_gamma = 0
                for t in range(len(sent)):
                    sum_gamma += gamma_state[t].get((cur_t, next_t), 0)
                    if sent[t] == word:
                        sum_gamma_obs += gamma_out[t].get((word, cur_t, next_t), 0)

                # Creating the entry if the value considered doesn't exist
                if not sout_p[2].get(cur_t):
                    sout_p[2][cur_t] = {}
                if not sout_p[2][cur_t].get(next_t):
                    sout_p[2][cur_t][next_t] = {}
                if not sout_p[2][cur_t][next_t].get(word):
                    sout_p[2][cur_t][next_t][word] = 0

                new_out_p = sum_gamma_obs / sum_gamma

                if abs(new_out_p - sout_p[2][cur_t][next_t][word]) > epsilon:
                    convergence -= 1

                sout_p[2][cur_t][next_t][word] = new_out_p

            if convergence > 0:
                break
        if convergence > 0:
            print('INFO: The Baum-Welch algorithm converged')
            break

    # Smoothing
    lambdas_out = HMM.em_algorithm(sout_p, heldout, type='output')
    smoothed_output_p = HMM.get_smoothed_output_probability(sout_p, lambdas_out)
    lambdas_trans = HMM.em_algorithm(strans_p, heldout, type='transition')
    smoothed_trans_p = HMM.get_smoothed_transition_probability(strans_p, lambdas_trans)

    return [smoothed_trans_p, smoothed_output_p]


def compute_csi_gamma(sentence, strans_p, alpha, beta, smoothed_output_p):
    """
    Computes the csi and the gammas for the update of the transition and output probabilities
    :param sentence: the data over which you need to compute the values
    :param strans_p: transition probability
    :param alpha: coefficients of the forward pass
    :param beta: coefficients of the backward pass
    :param smoothed_output_p: output probability
    :return:
    """
    csi = []
    gamma_out = []
    gamma_state = []

    # Estimate the parameters for updating the probabilities as described at slide 189.
    # We only perform one pass through the data
    for t in range(0, len(sentence) - 1):
        csi.append({})
        tot_csi = 0
        for next_t in strans_p[3].keys():
            for cur_t in strans_p[3][next_t].keys():
                for prev_t in strans_p[3][next_t][cur_t].keys():
                    if alpha[t].get((prev_t, cur_t)) and beta[t + 1].get((cur_t, next_t)):
                        # Considering sentence[t] and not sentence[t+1] because the index of the sentence is 0-based
                        emm_prob = smoothed_output_p[2].get(cur_t, {}).get(next_t, {}).get(sentence[t + 1], 0) \
                                   + smoothed_output_p[1].get(sentence[t], 0) \
                                   + smoothed_output_p[0]

                        # Computing transition probability
                        trans_p = strans_p[3].get(next_t, {}).get(cur_t, {}).get(prev_t, 0) \
                                  + strans_p[2].get(next_t, {}).get(cur_t, 0) \
                                  + strans_p[1].get(next_t, 0) \
                                  + strans_p[0]

                        # Getting a(s,t) * p(s'|s) * p(y_(t+1)|s') * beta(s',t+1)
                        # s = (prev, cur) ---- s' = (cur, next)
                        inc = alpha[t][prev_t, cur_t] * trans_p * emm_prob * beta[t + 1][
                            cur_t, next_t]

                        # Sum over time
                        csi[t][prev_t, cur_t, next_t] = inc
                        tot_csi += inc

        for k in csi[t].keys():
            csi[t][k] /= tot_csi

    for t in range(0, len(sentence)):
        tot_gamma = 0
        gamma_state.append({})
        gamma_out.append({})

        for prev_t, cur_t in alpha[t].keys():
            prod = alpha[t][prev_t, cur_t] * beta[t].get((prev_t, cur_t), 0)
            # Probability of being in a given state at any time and emitting a word
            gamma_out[t][sentence[t], prev_t, cur_t] = prod
            # Probability of being in a given state at any time
            gamma_state[t][prev_t, cur_t] = prod
            tot_gamma += prod

        for k in gamma_out[t].keys():
            gamma_out[t][k] /= tot_gamma
        for k in gamma_state[t].keys():
            gamma_state[t][k] /= tot_gamma

    return csi, gamma_out, gamma_state


def compute_alphas(sentence, tags, strans_p, sout_p):
    """
    Computes the alpha's (forward coefficients) over the data given in input

    :param sentence: Data over which the alphas are actually computed
    :param tags: Set of unique tags known
    :param strans_p: Smoothed transition probability
    :param sout_p: Smoothed output probability
    :return: Set of alphas, Normalization factors (needed when computing the betas)
    """
    alpha = [{}]
    initial_alpha = 1
    alpha[0]["<START>", "<START>"] = initial_alpha
    # Initializing a list with all the normalization factors so that you don't have to do the dynamical resizing of the
    # list while adding the normalization factors
    normalization_factors = [None] * len(sentence)
    normalization_factors[0] = 1

    for t in range(0, len(sentence) - 1):
        alpha.append({})
        # Based on the iteration, we have constraints on what the previous tags could be.
        # The tag "<START>" can only be at the beginning of the sentence. The fully connected graph is fully
        # connected only after the first 2 words
        if t == 0:
            prevprev_tags = ["<START>"]
            prev_tags = ["<START>"]
        elif t == 1:
            prevprev_tags = ["<START>"]
            prev_tags = tags
        else:
            prevprev_tags = prev_tags = tags

        # Keeps track of the alphas at the current step for normalization
        alpha_step_t = 0

        # For every current state possible
        for prev_t in prev_tags:
            for cur_t in tags:
                cur_alpha = 0
                # Get the emission probability
                emm_prob = sout_p[2].get(prev_t, {}).get(cur_t, {}).get(sentence[t + 1], 0) \
                           + sout_p[1].get(sentence[t], 0) \
                           + sout_p[0]

                for prevprev_t in prevprev_tags:
                    trans_p = strans_p[3].get(cur_t, {}).get(prev_t, {}).get(prevprev_t, 0) \
                              + strans_p[2].get(cur_t, {}).get(prev_t, 0) \
                              + strans_p[1].get(cur_t, 0) \
                              + strans_p[0]

                    cur_alpha += alpha[t].get((prevprev_t, prev_t), 0) \
                                * trans_p \
                                * emm_prob
                # If the current alpha is not zero, add it to the set of valid alphas (the ones not present are assumed
                # to be 0.
                if cur_alpha != 0:
                    alpha_step_t += cur_alpha
                    alpha[t + 1][prev_t, cur_t] = cur_alpha

        alpha[t + 1] = normalize(alpha[t + 1], alpha_step_t)
        normalization_factors[t + 1] = alpha_step_t

    return alpha, normalization_factors


def compute_betas(sentence, tags, strans_p, sout_p, normalization_factors):
    """
    Computes the betas in the backward pass going from right to left.
    The beta's at every step are normalized as well but the normalization is done with the same normalization values used
    for the alphas in the forward pass
    :param sentence: The data over which you need to compute the betas
    :param tags: The set of all tags seen
    :param strans_p: Smoothed transition probability
    :param sout_p: Smoothed output probability
    :param normalization_factors: The normalization factors computed in the forward pass
    :return: Returns the set of betas
    """
    beta = []
    for t in range(len(sentence)):
        beta.append({})

    initial_beta = 1
    # Initialize beta(x, T)
    for prev_t in tags:
        for cur_t in tags:
            beta[len(sentence) - 1][(prev_t, cur_t)] = initial_beta

    for t in range(len(sentence) - 2, -1, -1):
        #beta.append({})
        # Keeps of the alphas at the current step for normalization
        beta_step_t = 0
        if beta[t+1] == {}:
            continue
        # When reading the first word (so I reached the end of the string), the only previous tag is
        # "<START>", you're going from the state (<START>,t1) to (t1,t2)
        if t == 0 or t == 1:
            prev_tags = ["<START>"]
        else:
            prev_tags = tags
        if t == 0:
            cur_tags = ["<START>"]
        else:
            cur_tags = tags
        # For every current state possible
        for prev_t in prev_tags:
            for cur_t in cur_tags:
                cur_beta = 0
                for next_t in tags:
                    # Get the emission probability of the word in the next time stamp (i+1)
                    emm_prob = sout_p[2].get(cur_t, {}).get(next_t, {}).get(sentence[t + 1], 0) \
                               + sout_p[1].get(sentence[t], 0) \
                               + sout_p[0]

                    trans_p = strans_p[3].get(next_t,{}).get(cur_t, {}).get(prev_t, 0) \
                              + strans_p[2].get(next_t, {}).get(cur_t, 0) \
                              + strans_p[1].get(next_t, 0) \
                              + strans_p[0]

                    # Computing the current beta with the usual formula
                    cur_beta += beta[t + 1].get((cur_t, next_t), 0) \
                                * trans_p \
                                * emm_prob

                # Obtaining current beta and performing the normalization according to the normalization factor for
                # the corresponding alpha (as stated in slide 185)
                if cur_beta != 0:
                    beta_step_t += cur_beta
                    beta[t][prev_t, cur_t] = cur_beta / normalization_factors[t]
    return beta


def normalize(params, alpha_step_t):
    """
    Normalizes all the parameters by the factorization value provided in input
    :param params: Dictionary which values need to be normalized
    :param alpha_step_t: The normalization factor
    :return: A dictionary where all the values are normalized
    """
    # Dividing every entry by their corresponding normalization parameter
    for k in params.keys():
        params[k] = params[k] / alpha_step_t

    return params



def get_trans_unif_prob(tags):
    """
    Initializes the transition probabilities with uniform probabilities everywhere.

    NOTE: NOT USED, we are using the estimation of the parameters directly from the Language Modeling

    :param tags:
    :return:
    """
    trans_p = {}
    # Fully connected graph +
    # transitions from the initial state ("<START>","<START>") +
    # transitions from first word ("<START">,"tag1") -> ("tag1", "tag2")
    total_transitions = math.pow(len(tags), 3) + len(tags) + math.pow(len(tags), 2)

    uniform_prob = 1 / total_transitions
    for cur in tags:
        if not trans_p.get(cur):
            trans_p[cur] = {}
        for prev in tags:
            if not trans_p[cur].get(prev):
                trans_p[cur][prev] = {}
            for prevprev in tags:
                if not trans_p[cur][prev].get(prevprev):
                    trans_p[cur][prev][prevprev] = uniform_prob

    # Handling the tags at the beginning of the sentence
    for cur in tags:
        if not trans_p.get(cur):
            trans_p[cur] = {}
        for prev in tags:
            if not trans_p[cur].get(prev):
                trans_p[cur][prev] = {}
            trans_p[cur][prev]["<START>"] = uniform_prob
        trans_p[cur]["<START>"] = {}
        trans_p[cur]["<START>"]["<START>"] = uniform_prob

    return trans_p
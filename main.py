import math
import random
import statistics


def eval_conditional_prob(w):
    prob = {}
    for k in w.keys():
        prob[k] = {}
        tot = sum(w[k].values())
        for kk in w[k].keys():
            prob[k][kk] = w[k][kk] / tot

    return prob


def eval_joint_prob(w, data_cardinality):
    prob = {}
    for k in w.keys():
        prob[k] = {}
        for kk in w[k].keys():
            prob[k][kk] = w[k][kk] / data_cardinality

    return prob


def eval_cond_entropy(cond_prob, joint_prob):
    entropy = 0
    for k in cond_prob.keys():
        for kk in cond_prob[k].keys():
            entropy += joint_prob[k][kk] * math.log2(cond_prob[k][kk])

    return -entropy


FILENAME = "TEXTCZ1.txt"
prob_rand = [0, 0.1, 0.05, 0.01, 0.001, 0.0001]
f = open(FILENAME,"r")
letters = []
#Reading the file and picking all the letters from the alphabet used in the file
for x in f:
#Picking letters from the alphabet used
    for let in x:
        if let not in letters:
            letters.append(let)

#shuffling the letters at random
random.shuffle(letters)
f.close()


for p in prob_rand:
    print("\n\nRunning now the experiment with probability of changing letters: ", p, "\n")
    #Iterating only once if the input data don't get modified
    if p == 0:
        it = -9
    else:
        it = 0
    entropies = []
    perplexities = []
    for i in range(0,10 + it):
        f = open(FILENAME,"r")
        words = {}
        count_words = 0
        prev = ""   #First character to treat also "<>,<1st_word>" in the same way as other pairs
        #For each word in the file
        for x in f:
            x = x[:-1]
            
            #Iterate through the letters of the given and word and randomly change letters
            for idx, el in enumerate(x):
                if random.uniform(0, 1) <= p:
                    x = x[:idx] + random.choice(letters) + x[idx + 1:]

            if not words.get(prev):
                words[prev] = {}
            if not words[prev].get(x):
                words[prev][x] = 0

            #Adding 1 to the number of the occurences (prev, x), in this order
            words[prev][x] += 1
            prev = x
            count_words += 1

        ## EVALUATION OF CONDITIONAL ENTROPY

        cond_prob = eval_conditional_prob(words)
        joint_prob = eval_joint_prob(words, count_words)

        cond_entr = eval_cond_entropy(cond_prob, joint_prob)

        entropies.append(cond_entr)


        ## EVALUATION OF PERPLEXITY

        perplexity = math.pow(2,cond_entr)

        perplexities.append(perplexity)

    print("CONDITIONAL ENTROPY:")
    print("min -> ", min(entropies))
    print("max -> ", max(entropies))
    print("avg -> ", statistics.mean(entropies))

    print("\nPERPLEXITY:")
    print("min -> ", min(perplexities))
    print("max -> ", max(perplexities))
    print("avg -> ", statistics.mean(perplexities))

for p in prob_rand[1:]:
    print("\n\nRunning now the experiment with probability of mixing words: ", p, "\n")
    entropies = []
    perplexities = []
    original_words = []
        for x in f:
            x = x[:-1]
            original_words.append(x)
            count_words += 1
            
    for i in range(0, 10):
        f = open(FILENAME, "r")
        list_words = original_words.copy()

        #Switching the position of the words inside the file
        for (index,w) in enumerate(list_words):
            if random.uniform(0,1) < p:
                ind = random.randint(0, len(list_words) - 1)
                tmp = list_words[ind]
                list_words[ind] = list_words[index]
                list_words[index] = tmp

        words = {}
        prev = ""   #First character to treat also "<>,WHEN" in the same way as other pairs
        for x in list_words:
            x = x[:-1]

            if not words.get(prev):
                words[prev] = {}
            if not words[prev].get(x):
                words[prev][x] = 0

            words[prev][x] += 1
            prev = x

        ## EVALUATION OF CONDITIONAL ENTROPY

        cond_prob = eval_conditional_prob(words)
        joint_prob = eval_joint_prob(words, count_words)

        cond_entr = eval_cond_entropy(cond_prob, joint_prob)

        entropies.append(cond_entr)


        ## EVALUATION OF PERPLEXITY

        perplexity = math.pow(2,cond_entr)

        perplexities.append(perplexity)

    print("CONDITIONAL ENTROPY:")
    print("min -> ", min(entropies))
    print("max -> ", max(entropies))
    print("avg -> ", statistics.mean(entropies))

    print("\nPERPLEXITY:")
    print("min -> ", min(perplexities))
    print("max -> ", max(perplexities))
    print("avg -> ", statistics.mean(perplexities))
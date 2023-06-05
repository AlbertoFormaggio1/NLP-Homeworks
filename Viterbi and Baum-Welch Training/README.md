Use the datasets T, H, and S. Estimate the parameters of an HMM tagger using supervised learning off the T data (trigram and lower models for tags). Smooth (both the trigram tag model as well as the lexical model) in the same way as in Homework No. 1 (use data H). Evaluate your tagger on S, using the Viterbi algorithm.

Now use only the first 10,000 words of T to estimate the initial (raw) parameters of the HMM tagging model. Strip off the tags from the remaining data T. Use the Baum-Welch algorithm to improve on the initial parameters. Smooth as usual. Evaluate your unsupervised HMM tagger and compare the results to the supervised HMM tagger.

Tabulate and compare the results of the HMM tagger vs. the Brill's tagger.

This task will show you the importance of smoothing for language modeling, and in certain detail it lets you feel its effects.

First, you will have to prepare data: take the same texts as in the previous task, i.e.

[TEXTEN1.txt](TEXTEN1.txt) and [TEXTCZ1.txt](TEXTCZ1.txt)

Prepare 3 datasets out of each: strip off the last 20,000 words and call them the Test Data, then take off the last 40,000 words from what remains, and call them the Heldout Data, and call the remaining data the Training Data.

Here comes the coding: extract word counts from the training data so that you are ready to compute unigram-, bigram- and trigram-based probabilities from them; compute also the uniform probability based on the vocabulary size. Remember ( $T$ being the text size, and $V$ the vocabulary size, i.e. the number of types - different word forms found in the training text):
$$p_0(w_i)=1/V$$
$$p_1(w_i)=c_1(w_i)$$
$$p_2(w_i|w_{i-1})=c_2(w_i,w_{i-1})/c_1(w_{i-1})$$
$$p_3(w_i|w_{i-1},w_{i-2}) = c_3(w_i,w_{i-1},w_{i-2})/c_2(w_{i-1},w_{i-2})$$

Be careful; remember how to handle correctly the beginning and end of the training data with respect to bigram and trigram counts.

Now compute the four smoothing parameters (i.e. "coefficients", "weights", "lambdas", "interpolation parameters", for the trigram, bigram, unigram and uniform distributions) from the heldout data using the EM algorithm. (Then do the same using the training data again: what smoothing coefficients have you got? After answering this question, throw them away!) Remember, the smoothed model has the following form:
$$p_s(w_i|w_{i-2},w_{i-1})=\lambda_0p_0(w_i)+\lambda_1p_1(w_i)+\lambda_2p_2(w_i|w_{i-1})+\lambda_3p_3(w_i|w_{i-1},w_{i-2})$$

where

$$\sum_{i=0}^3\lambda_i=1$$

And finally, compute the cross-entropy of the test data using your newly built, smoothed language model. Now tweak the smoothing parameters in the following way: add 10%, 20%, 30%, ..., 90%, 95% and 99% of the difference between the trigram smoothing parameter and 1.0 to its value, discounting at the same the remaining three parameters proportionally (remember, they have to sum up to 1.0!!). Then set the trigram smoothing parameter to 90%, 80%, 70%, ... 10%, 0% of its value, boosting proportionally the other three parameters, again to sum up to one. Compute the cross-entropy on the test data for all these 22 cases (original + 11 trigram parameter increase + 10 trigram smoothing parameter decrease). Tabulate, graph and explain what you have got.

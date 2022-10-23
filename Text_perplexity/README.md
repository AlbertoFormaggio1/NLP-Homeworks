# Entropy of a text

<p>In this experiment, you will determine the conditional entropy of the word distribution in a text given the previous word. To do this, you will first have to compute $P(i,j)$ which is the probability that at any position in the text you will find the word i followed immediately by the word j, and $P(j|i)$, which is the probability that if word i occurs in the text then word j will follow. Given these probabilities, the conditional entropy of the word distribution in a text given the previous word can then be computed as:</p>
$$H(J|I)=- \sum_{i\in I,j\in J} P(i,j) log_2(P(i|j))$$
<p>The perplexity is then computed simply as:</p>
$$PX(P(I|J)) = 2^{H(J|I)}$$
<p>Compute this conditional entropy and perplexity for the file</p>
<p><strong><a href="./TEXTEN1.txt">TEXTEN1.txt</a></strong></p>
<p>This file has every word on a separate line. (Punctuation is considered a word, as in many other cases.) The i,j above will also span sentence boundaries, where i is the last word of one sentence and j is the first word of the following sentence (but obviously, there will be a fullstop at the end of most sentences).</p>
<p>Next, you will mess up the text and measure how this alters the conditional entropy. For every character in the text, mess it up with a likelihood of 10%. If a character is chosen to be messed up, map it into a randomly chosen character from the set of characters that appear in the text. Since there is some randomness to the outcome of the experiment, run the experiment 10 times, each time measuring the conditional entropy of the resulting text, and give the min, max, and average entropy from these experiments. Be sure to use srand to reset the random number generator seed each time you run it. Also, be sure each time you are messing up the original text, and not a previously messed up text. Do the same experiment for mess up likelihoods of 5%, 1%, .1%, .01%, and .001%.</p>
<p>Next, for every word in the text, mess it up with a likelihood of 10%. If a word is chosen to be messed up, map it into a randomly chosen word from the set of words that appear in the text. Again run the experiment 10 times, each time measuring the conditional entropy of the resulting text, and give the min, max, and average entropy from these experiments. Do the same experiment for mess up likelihoods of 5%, 1%, .1%, .01%, and .001%.</p>
<p>Now do exactly the same for the file</p>
<p><strong><a href="./TEXTCZ1.txt">TEXTCZ1.txt</a></strong></p>
<p>which contains a similar amount of text in an unknown language (<em>just FYI, that's Czech</em> [*])</p>
<p>Tabulate, graph and explain your results. Also try to explain the differences between the two languages. To substantiate your explanations, you might want to tabulate also the basic characteristics of the two texts, such as the word count, number of characters (total, per word), the frequency of the most frequent words, the number of words with frequency 1, etc.</p>
<p>Attach your source code commented in such a way that it is sufficient to read the comments to understand what you have done and how you have done it.</p>

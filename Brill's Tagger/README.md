# Brill's Tagger and Tagger Evaluation

Download Eric Brill's supervised tagger from UFAL's course assignment space. Install it (i.e., uncompress (gunzip), untar, and make). You might need to make some changes in his makefile of course (it's and OLD program, in this fast changing world...).

Alternativelly, you can use the NLTK implementation in Python (pip install --user -U nltk).

After installation, get the data, train it on as much data from T as time allows (in the package, there is an extensive documentation on how to train it on new data), and evaluate on data S. Tabulate the results.

Do cross-validation of the results: split the data into S', [H',] T' such that S' is the first 40,000 words, and T' is the last but the first 20,000 words from the rest. Train Eric Brill's tagger on T' (again, use as much data as time allows) and evaluate on S'. Again, tabulate the results.

Do three more splits of your data (using the same formula: 40k/20k/the rest) in some way or another (as different as possible), and get another three sets of results. Compute the mean (average) accuracy and the standard deviation of the accuracy. Tabulate all results.

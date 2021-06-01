for organized results please refer to:
https://docs.google.com/spreadsheets/d/1FwFI3Qputja9rU8H_0axmd91ZxgDF9hTuxI3ecPhkSc/edit#gid=0
# NLP_ex2
# Results:
# MLE_4b_basic error rate (basic MLE):
Total  error rate 0.14806309904153359

Seen   error rate 0.07044634806131655
 
Unseen error rate 0.75

# HMM_4c_test error rate (viterbi HMM no smoothing):
Total  error rate 0.7172523961661341

Seen   error rate 0.7130297565374211

Unseen error rate 0.75

# HMM_4d_test error rate(viterbi HMM with smoothing):
Total  error rate 0.21465654952076674

Seen   error rate 0.15024797114517585

Unseen error rate 0.7141608391608392

# e1 error rate (pw no smoothing):
Total  error rate 0.19219249201277955

Seen   error rate 0.15757439134355278

Unseen error rate 0.4606643356643356

# e2 error rate (pw with smoothing):
Total  error rate 0.1905950479233227

Seen   error rate 0.15013525698827768

Unseen error rate 0.5043706293706294


# CONCLUSIONS:
The HMM Algorithm suffered from sparsity, and performed very poorly,

If we're giving zero probabilities to the unseen words and tags(with no smoothing). 

With the add one smoothing we improved it. 

With psuedo words we also improved on the unseen words by about 25-30%.


# Comment on Confusion Matrix and common Errors:
the Confusion Matrix shows that the HMM with PW and smoothing in e2, made most of his errors between

the NN, NP, NNS, JJ tags. and between VBD, VBN:

JJ  (usually predicted NN or NP).

NN  (usually predicted JJ or NP).

NNS (usually predicted NN, JJ or NP).

and errors such as prediction VBD instead VBN and vice versa.

The document e2_confusion_mat.xlsx is attached.




# Side note (NOT REQUIRED) 
# HMM error rate - without zero probability un-normalized
Total  error rate 0.12250399361022368

Seen   error rate 0.04779080252479706

Unseen error rate 0.7019230769230769

Even though it was not required:
Some thing we've noticed is that the best prediction (about 4.8% error rate)
happened while we were running the basic HMM algorithm but instead of returning zero
in the unseen case we returned small non-zero value.
Even though its result is not a proper probability function (values won't sum to 1)

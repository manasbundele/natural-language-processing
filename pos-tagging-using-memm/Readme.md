# Part-of-Speech tagging using Maximum Entropy Markov Model with Viterbi decoding

In this project, we predict the part of speech tag for a word in a given sentence. The training is done using the Universal tagset in Brown Corpus with 12 different tags.

## Implementation Details

1. Data: Brown corpus with Universal tagset as training data
2. Feature generation
  * Word Ngram features
  * Word specific features 
3. Removal of rare features and training a Maximum Entropy Markov Model
4. Viterbi Decoding to get the highest-probability sequence of tags for
each test sentence


## Results of POS tagging

Sentence: ['Apple', 'Inc.', 'is', 'an', 'American', 'multinational', 'technology', 'company', 'headquartered', 'in', 'Cupertino', ',', 'California', '.']

>Tags: [u'NOUN', u'NOUN', u'VERB', u'DET', u'ADJ', u'ADJ', u'NOUN', u'NOUN', u'VERB', u'ADP', u'NOUN', u'.', u'NOUN', u'.']


Sentence: ['The', 'grand', 'jury', 'commented', 'on', 'a', 'number', 'of', 'other', 'topics', '.']

>Tags: [u'DET', u'ADJ', u'NOUN', u'VERB', u'ADP', u'DET', u'NOUN', u'ADP', u'ADJ', u'NOUN', u'.']


Sentence: ['Congress', 'is', 'in', 'a', 'standoff', 'with', 'the', 'Trump', 'administration', 'over', 'its', 'refusal', 'to', 'share', 'a', 'whistle-blower', 'complaint', 'with', 'lawmakers', '.']

>Tags: [u'NOUN', u'VERB', u'ADP', u'DET', u'NOUN', u'ADP', u'DET', u'NOUN', u'NOUN', u'ADP', u'DET', u'NOUN', u'PRT', u'VERB', u'DET', u'ADJ', u'NOUN', u'ADP', u'NOUN', u'.']


Sentence: ['I', 'want', 'to', 'go', 'to', 'a', 'restaurant', '.']

>Tags: [u'PRON', u'VERB', u'PRT', u'VERB', u'ADP', u'DET', u'NOUN', u'.']


sentence = ['IBM', 'was', 'founded', 'in', '1924', '.']

>tags= [u'NOUN', u'VERB', u'VERB', u'ADP', u'NUM', u'.']

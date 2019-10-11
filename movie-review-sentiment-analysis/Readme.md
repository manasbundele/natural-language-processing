# Sentiment Analysis of Movie Reviews

This project consists of training a Naive Bayes and Logistic Regression classifier to predict the sentiment of a review. It also uses a sentiment lexicon, Dictionary of Affect in Language, which is further augmented using WordNet to further improve sentiment analysis results.

### Implementation steps

1. Tokenization and Preprocessing
  * Identify single quotes suing regex and split them
  * Tagging words in [square backets] with prefix 'EDIT_' (The words inside the square brackets weren't written by
the original author, but were added by an editor to clarify the meaning of the snippet)
  * Negation Tagging
  
2. Naive Bayes and Logistic Regression Classifier
3. Using a Lexicon with metrics for each word:
  * Activeness, the level of activation or arousal of a word (eg. "sleep" vs. "run")
  * Evaluation, the pleasantness of a word (eg. "happy" vs. "sad")
  * Imagery, the concreteness of a word (eg. "ower" vs. "freedom")
4. Adding support for synonyms and antonyms to DAL using WordNet


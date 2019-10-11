#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: manasbundele
"""

import numpy as np
import nltk
nltk.download('brown')
nltk.download('universal_tagset')
from nltk.corpus import brown
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
import pickle

rare_words = set()
feature_dict = dict()
tag_dict = dict()

# change this to 0 if you dont want to add features in extra credit section
# e.g. word-[word]-prevtag-[prevtag], allcaps, 
# ‘wordshape-[x]’, where [x] is an abstracted version of the word where lowercase letters are replaced with ‘x’, uppercase letters are replaced with ‘X’, and digits are replaced with ‘d’. Note that word shape features should not be lowercased, unlike the other features.
# ‘short-wordshape-[x]’, like the normal word shape feature, except consecutive character types are merged into a single character (eg. “hello” → ‘xxxxx’ → ‘x’ and “Hello” → ‘Xxxxx’ → ‘Xx’). Again, short word shape features should not be lowercased.
# ‘allcaps-digit-hyphen’
# ‘capital-followedby-co’, for words that are capitalized and where the words “Co.” or “Inc.” occur at most three words after the word.
EXTRA_CREDIT = 0


def find_word_count(list_of_sentences):
    ''' This function finds count of each word in corpus '''
    word_count = dict()
    
    for sentence in list_of_sentences:
        for word in sentence:
            if word_count.get(word) == None:
                word_count[word] = 1
            else:
                word_count[word] = word_count[word] + 1
                
    return word_count
    

def find_rare_words(word_count_data, count):
    ''' This function finds rare words based on data in word count dict'''
    rare = set([k for k,v in word_count_data.items() if v<count])
    
    return rare    



def word_ngram_features(i, words):
    ''' this function returns a list of ngram features for a given word '''
    # start and end padding
    padded_words = list(words)
    
    padded_words.insert(0,'<s>')
    padded_words.insert(0,'<s>')
    padded_words.extend(['</s>','</s>'])
    
    new_i = i+2 # due to start padding
     
    prevbigram = 'prevbigram-' + padded_words[new_i-1]
    nextbigram = 'nextbigram-' + padded_words[new_i+1]
    prevskip = 'prevskip-' + padded_words[new_i-2]
    nextskip = 'nextskip-' + padded_words[new_i+2]
    prevtrigram = 'prevtrigram-' + padded_words[new_i-1] + '-' + padded_words[new_i-2]
    nexttrigram = 'nexttrigram-' + padded_words[new_i+1] + '-' + padded_words[new_i+2]
    centertrigram = 'centertrigram-' + padded_words[new_i-1] + '-' + padded_words[new_i+1] 
    
    return [prevbigram, nextbigram, prevskip, nextskip, prevtrigram, nexttrigram, centertrigram]


def word_features(word, rare_words):
    ''' this function return a list of features related to a word - word in rare words?, capital? number? 
        does it have hyphen? 1 to 4 character prefixes and suffixes
    
    '''
    
    features  = []
    
    if word not in rare_words:
        features.append('word-'+word)
    
    if word[0].isupper():
        features.append('capital')
        
    if any(char.isdigit() for char in word):
        features.append('number')
        
    if '-' in word:
        features.append('hyphen')
        
        
    word_len = len(word)
    minimum_chars = min(word_len, 4)
    
    for i in range(minimum_chars):
        prefix = word[:i+1]
        features.append('prefix'+str(i+1)+'-'+prefix)
        
        suffix = word[word_len-i-1:]
        features.append('suffix'+str(i+1)+'-'+suffix)
    
    return features 


def get_features(i, words, prevtag, rare_words):
    
    features = word_features(words[i], rare_words)
    ngram_features = word_ngram_features(i, words)
    
    all_features = features + ngram_features
    
    # adding previous word's tag 
    all_features.append('tagbigram-'+prevtag)
    
    # adding for part 6: extra credit, more features in extra credit
    if EXTRA_CREDIT == 1:
        all_features.append('word-'+words[i]+'prevtag-'+prevtag)
        
        if words[i].isupper():
            all_features.append('allcaps')
            
        digit_flag = 0 # if digit in word
        
        # creating string as XXxd for word like MMm1
        abstracted_str = ''
        for char in words[i]:
            if char.isupper():
                abstracted_str = abstracted_str + 'X'
            elif char.islower():
                abstracted_str = abstracted_str + 'x'
            elif char.isdigit():
                abstracted_str = abstracted_str + 'd'
                digit_flag = 1
            
        all_features.append('wordshape-'+abstracted_str)
        
        # for short-wordshape -> remove multiple consecutive chars from abstracted string
        short_wordshape = ''
        templen=len(abstracted_str)
        for idx in range(templen):
            #print idx, idx+1, templen-1, abstracted_str[idx]
            if idx+1 <= templen - 1 and abstracted_str[idx] != abstracted_str[idx+1]:
                short_wordshape = short_wordshape + abstracted_str[idx]
            elif idx+1 > templen - 1 and (len(short_wordshape) != 0 and short_wordshape[-1] != abstracted_str[idx]) or len(short_wordshape) == 0:
                short_wordshape = short_wordshape + abstracted_str[idx]
            
        all_features.append('short-wordshape-'+short_wordshape)
        
        # if word is all upercase and has at least one digit and one hyphen
        if words[i].isupper() and digit_flag == 1 and '-' in words[i]:
            all_features.append('allcaps-digit-hyphen')
        
        # if word is capitalized and has Inc or Co in upto next 3 words
        if words[i][0].isupper():
            for k in range(3):
                if i+k+1 < len(words) and (words[i+k+1] == 'Co.' or words[i+k+1] == 'Inc.'):
                    all_features.append('capital-followedby-co')
    
    # lowercase only if wordshape not in it as we dont want to lower wordshape or shortwordshape
    for k in range(len(all_features)):
        if 'wordshape' not in all_features[k]:
            all_features[k] = all_features[k].lower() 
    
    return all_features
    
    
def remove_rare_features(features, n):
    feature_count = dict()
    
    # getting feature count
    for sentence_data in features:
        for word_data in sentence_data:
            for feature in word_data:
                if feature_count.get(feature) == None:
                    feature_count[feature] = 1
                else:
                    feature_count[feature] = feature_count[feature] + 1
    
    rare_features = set()
    non_rare_features = set()
   
    # finding rare and non rare features
    for k,v in feature_count.items():
        if v < n:
            rare_features.add(k)
        else:
            non_rare_features.add(k)
            
    new_features = []
    # removing rare features
    for sentence in features:
        s = []
        for word_data in sentence:
            w = []
            for feature in word_data:
                if feature not in rare_features:
                    w.append(feature)
            s.append(w)
        new_features.append(s)

    return new_features, non_rare_features
    
def build_Y(tags):
    ''' this function creates a single list of tags corresponding to each training example'''
    Y = []
    
    for s_tag in tags:
        for w_tag in s_tag:
            Y.append(tag_dict[w_tag])
             
    return np.array(Y)


def build_X(training_features):
    ''' this function creates a matrix of size (no. of training examples * feature dictionary length)'''
    examples = []
    features = []
    
    i=0
    # creating two lists examples and features to make csr matrix
    for sentence_data in training_features:
        for word_data in sentence_data:
            for feature in word_data:
                if feature in feature_dict:
                    examples.append(i)
                    features.append(feature_dict[feature])
            i = i + 1
    
    # value is 1 for row,col values present in examples and features
    values = [1] * len(examples)
    
    examples = np.array(examples)
    features = np.array(features)
    values = np.array(values)
    
    return csr_matrix((values, (examples, features)),shape=(i,len(feature_dict)))

def load_test(filename):
    output = []
    with open(filename,'r') as f:
        for line in f:
            words = [word.rstrip('\n') for word in line.split(" ") if word]
            output.append(words)
            
    return output


def get_predictions(test_sentence, model):
    # Y_pred is a matrix with n-1 * T *T dimension
    Y_pred = np.zeros((len(test_sentence[0])-1, len(tag_dict), len(tag_dict)))   

    for i in range(len(test_sentence[0])):
        # we skip first word because we create Y_start for it
        if i ==0:
            continue
        j=0
        for key, value in sorted(tag_dict.items(), key=lambda kv: kv[1]):
            features = get_features(i, test_sentence[0], key, rare_words)
            X = build_X([[features]])
            Y_pred[i-1][j] = model.predict_log_proba(X)
            j = j + 1

    # for first word prev tag is <S>, we create Y_start here
    first_word_feature = get_features(0, test_sentence[0], '<S>', rare_words)
    first_word_X = build_X([[first_word_feature]])
    Y_start = np.array(model.predict_log_proba(first_word_X))
    
    return Y_pred, Y_start
         

def viterbi(Y_start, Y_pred):
    sentence_len = Y_pred.shape[0]+1
    tag_len = len(tag_dict)
    
    V = np.zeros((sentence_len, tag_len))
    BP = np.zeros((sentence_len, tag_len), dtype=int)
    
    # base case
    for j in range(tag_len):
        V[0][j] = Y_start[0][j]
        BP[0][j] = -1

    # building dp matrix
    for i in range(sentence_len - 1):
        for k in range(tag_len):
            temp = []
            for j in range(tag_len):
                temp.append(V[i][j] + Y_pred[i][j][k])
                
            V[i+1][k] = max(temp)
            BP[i+1][k] = np.argmax(temp)
            
    
    backward_indices = []
    # Y_pred.shape[0] is n-1, i.e. sentence length - 1
    n = sentence_len - 1
    # this is the last word with maximum combined probability
    index = np.argmax(V[n])
    backward_indices.append(index)
    
    # finding backward pointers to the tag that was generated for last word
    for idx in range(sentence_len - 1):
        n = sentence_len - 1 - idx
        index = BP[n][index]
        backward_indices.append(index)
        
    backward_indices.reverse()
    
    reversed_tag_dict = {v:k for k,v in tag_dict.items()}
    
    # decoding tag indices to keywords
    list_of_tags = []
    for tagidx in backward_indices:
        list_of_tags.append(reversed_tag_dict[tagidx])
        
    return list_of_tags


        
def main():
    brown_sentences = brown.tagged_sents(tagset='universal')
    
    train_sentences = []
    train_tags = []
    
    # sepearating sentences and their labels
    for sentence in brown_sentences:
        s = []
        t = []
        for pair in sentence:
            s.append(pair[0])
            t.append(pair[1])
        
        train_sentences.append(s)
        train_tags.append(t)
        
    # finding word count
    word_count = find_word_count(train_sentences)
    
    # finding rare words
    rare_words = find_rare_words(word_count, 5)
    
    # adding features for each word
    training_features = []
    for idx in range(len(train_sentences)):
        features = []
        sentence = train_sentences[idx]
        for i in range(len(sentence)):
            prevtag = '<S>' if i == 0 else train_tags[idx][i-1]
            features.append(get_features(i, sentence, prevtag, rare_words))
        training_features.append(features)
    
    # overwriting training features with values after removing rare features
    training_features, non_rare_features = remove_rare_features(training_features, 5)
    
    # creating feature dictionary
    counter = 0
    for feature in non_rare_features:
        feature_dict[feature] = counter
        counter = counter + 1
        
    # creating tag dictionary
    tag_counter = 0
    for sent_tags in train_tags:
        for tag in sent_tags:
            if tag not in tag_dict:
                tag_dict[tag] = tag_counter
                tag_counter= tag_counter + 1
    
    
    X_train = build_X(training_features)
    Y_train = build_Y(train_tags)
    
    print "X_train Y_train built"
    
    '''
    # if we want to save model then use this code
    filename = 'lr_model.sav'
    
    lr = pickle.load(open(filename, 'rb'))
    if lr == None:
        lr = LogisticRegression(class_weight='balanced', solver='saga', multi_class='multinomial', verbose=2)
        lr.fit(X_train, Y_train)
        
        print "Model fit"
        
        # save the model to disk
        pickle.dump(lr, open(filename, 'wb'))
    
    '''
    
    lr = LogisticRegression(class_weight='balanced', solver='saga', multi_class='multinomial', verbose=2)
    lr.fit(X_train, Y_train)
    
    print "Model fit"

    
    test_data = load_test('test.txt')
    # tag prediction
    for sentence in test_data:
        Y_pred, Y_start = get_predictions([sentence], lr)
        tags = viterbi(Y_start, Y_pred)
        print "sentence =", sentence
        print "tags=", tags
        print "\n"
        
        

if __name__ == "__main__":
    main()
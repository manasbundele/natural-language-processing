#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: manasbundele
"""

import nltk
import numpy as np
import re
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from nltk.corpus import wordnet


multiple_words_single_quote_regex = re.compile(r"(')(\w+)([^']+?)(\w+)(')")

single_word_single_quote_regex = re.compile("(')(\w+)(')")

regex_for_negation_words = re.compile(r"\b[Nn]ot\b|\b[Nn]o\b|\b[Nn]ever\b|\bcannot\b|(\w+n't)") 

# storing in a set
negation_ending_tokens = {'but','however','nevertheless','.','?','!'}

feature_dictionary = {}
dal_dict = {}
# to enable semantic lexicon DAL, set it to 1 : Part 4 and 6
DAL_FLAG=1


def load_corpus(corpus_path):
    
    text_file = open(corpus_path, "r")
    
    lines = text_file.read().splitlines()
    
    list_of_tuples = [(str(line.split("\t")[0]), int(line.split("\t")[1])) for line in lines]
    
    text_file.close()

    return list_of_tuples    
    
    

def tokenize(snippet):
    '''
    This function takes a snippet, separates out quotes and returns list of tokens
    '''
    # call single word first otherwise multiple_words_single_quote will split 'word' into chars
    preprocess1 = re.sub(single_word_single_quote_regex, r'\1 \2 \3', snippet)
    preprocess2 = re.sub(multiple_words_single_quote_regex, r'\1 \2\3\4 \5', preprocess1)
    
    tokens = preprocess2.split(" ")
    #print tokens
    return tokens


def tag_edits(tokenized_snippet):
    '''
    This function tags words in [] brackets(not written by authot but editor) with EDIT_
    '''
    mark_edit_flag = 0
    for i in range(len(tokenized_snippet)):
        first_char = tokenized_snippet[i][0]
        last_char = tokenized_snippet[i][-1]
        
        if first_char == "[":
            mark_edit_flag = 1
            if tokenized_snippet[i][1:] != '':
                tokenized_snippet[i] = tokenized_snippet[i][1:]
        
        if mark_edit_flag == 1:
            if tokenized_snippet[i] != '':
                tokenized_snippet[i] = "EDIT_" + tokenized_snippet[i]
        
        if last_char == "]":
            mark_edit_flag = 0
            # handling empty string case, since we are already appending EDIT_, checking edit for empty string case
            if tokenized_snippet[i][:-1] != 'EDIT_':
                tokenized_snippet[i] = tokenized_snippet[i][:-1]
            
    return tokenized_snippet
            
    
    
def tag_negation(tokenized_snippet):
    '''
    THis function tags negation words with NOT_
    '''
    # copy has edit tags removed
    tokenized_snippet_copy = [re.sub("EDIT_","", token) for token in tokenized_snippet]
    
    #print tokenized_snippet_copy
    tokenized_snippet_copy_with_pos = nltk.pos_tag(tokenized_snippet_copy)
    
    for i in range(len(tokenized_snippet_copy)):
        tokenized_snippet_copy_with_pos[i] = list(tokenized_snippet_copy_with_pos[i])
        tokenized_snippet_copy_with_pos[i][0] = tokenized_snippet[i]
        tokenized_snippet_copy_with_pos[i] = tuple(tokenized_snippet_copy_with_pos[i])
        
    negation_start_flag = 0
    
    list_length = len(tokenized_snippet_copy_with_pos) 
    
    for i in range(list_length):
        check_neg = regex_for_negation_words.search(tokenized_snippet_copy_with_pos[i][0])
        
        if check_neg != None:
            negation_start_flag = 1
        
        if negation_start_flag == 1:
            if tokenized_snippet_copy_with_pos[i][0] == 'not' and (i+1 < list_length) and tokenized_snippet_copy_with_pos[i+1][0] == 'only':
                negation_start_flag = 0
            else:
                tokenized_snippet_copy_with_pos[i] = list(tokenized_snippet_copy_with_pos[i])
                
                if re.sub("EDIT_","",tokenized_snippet_copy_with_pos[i][0]) in negation_ending_tokens:
                    negation_start_flag = 0
                elif tokenized_snippet_copy_with_pos[i][1] in ['JJR','RBR']:
                    negation_start_flag = 0
                else:
                    tokenized_snippet_copy_with_pos[i][0] = "NOT_" + tokenized_snippet_copy_with_pos[i][0]
                
                tokenized_snippet_copy_with_pos[i] = tuple(tokenized_snippet_copy_with_pos[i])
    
    return tokenized_snippet_copy_with_pos
                

def get_features(preprocessed_snippet):
    V = len(feature_dictionary)
    
    # as V already has dal mterics: activeness, pleasantness, imagery if FLAG is true
    feature = np.zeros(V)
    
    for i in range(len(preprocessed_snippet)):
        word = preprocessed_snippet[i][0]
        
        # skip OOV words in test data
        if "EDIT_" not in word and word in feature_dictionary:
            index = feature_dictionary[word]
            feature[index] = feature[index] + 1
            
    if DAL_FLAG:
        (activeness, pleasantness, imagery) = score_snippet(preprocessed_snippet, dal_dict)  
        
        feature[V-3] = activeness
        feature[V-2] = pleasantness
        feature[V-1] = imagery
        
    return feature
    
    
def normalize(X):
    max_arr =  np.amax(X, axis=0)
    min_arr = np.amin(X,axis=0)
    
    rows,cols = X.shape 
    
    for i in range(rows):
        for j in range(cols):
            # if denominator is not 0
            if (max_arr[j] - min_arr[j]) != 0:
                X[i][j] = (X[i][j] - min_arr[j])/float((max_arr[j] - min_arr[j]))
    return X


def evaluate_predictions(Y_pred, Y_true):
    tp = 0.0
    fp = 0.0
    fn = 0.0
    
    for i in range(len(Y_pred)):
        if Y_true[i] == 1 and Y_pred[i] == 1:
            tp = tp + 1
        elif Y_true[i] == 0 and Y_pred[i] == 1:
            fp = fp + 1
        elif Y_true[i] == 1 and Y_pred[i] == 0:
            fn = fn + 1
            
    precision = tp/float(tp+fp)
    
    recall = tp/float(tp+fn)
    
    fmeasure = (2 * precision * recall)/float(precision + recall)
    
    return (precision, recall, fmeasure)
    

def top_features(logreg_model, k):
    '''
    This function returns top k features for logistic regression
    '''
    weight_vec = logreg_model.coef_
    
    list_of_weight_tuples = [(index, weight_vec[0][index]) for index in range(len(weight_vec[0]))]
    
    sorted_list_of_weight_tuples = sorted(list_of_weight_tuples, key=lambda x: abs(x[1]), reverse=True)
    
    top_k_features = sorted_list_of_weight_tuples[:k]
    
    for i in range(k):
        top_k_features[i] = list(top_k_features[i])
        top_k_features[i][0] = [key for key,val in feature_dictionary.items() if val == top_k_features[i][0]][0]
        top_k_features[i] = tuple(top_k_features[i])
        
    print "Top ", k, " features:", top_k_features
        
    return top_k_features
    

def load_dal(dal_path):
    dal_file = open(dal_path, "r")
    
    lines = dal_file.read().splitlines()
    
    for i in range(len(lines)):
        if i==0:
            continue

        tokens = lines[i].split("\t")
        if tokens[0] not in dal_dict:
            dal_dict[tokens[0]] = (float(tokens[1]), float(tokens[2]), float(tokens[3]))
    
    dal_file.close()
    
    return dal_dict


def score_snippet(preprocessed_snippet, dal):
    '''
    This function scores the activeness, pleasantness and imagery values for snippet by averaging values for all words
    '''
    activeness = 0.0
    pleasantness = 0.0
    imagery = 0.0
    # This keeps count of how many words are found in dal dict so as to find average
    count = 0 
    for tup in preprocessed_snippet:
        if "EDIT_" in tup[0]: 
            continue
        elif "NOT_" in tup[0] and tup[0] in dal:
            # if not is present then all values are inverted
            activeness = activeness + (-1)* dal[tup[0]][0]
            pleasantness = pleasantness + (-1)* dal[tup[0]][1]
            imagery = imagery + (-1)* dal[tup[0]][2]
            count = count + 1
        elif tup[0] in dal:
            activeness = activeness + dal[tup[0]][0]
            pleasantness = pleasantness + dal[tup[0]][1]
            imagery = imagery + dal[tup[0]][2]
            count = count + 1
        # comment this else part if you want to get output till part 5
        
        else:
            # Part 6: look for synonym and antonym if word with dal metrics not found
            try:
                synonyms = set([])
                antonyms = set([])
                synonym_found = 0 # to check if we need to run it for antonym
                for syn in wordnet.synsets(tup[0], pos=get_wordnet_pos(tup[1])):
                    for lemma in syn.lemmas():
                        synonyms.add(lemma.name())
                        if lemma.antonyms():
                            antonyms.add(lemma.antonyms()[0].name())
                            
                for synonym in synonyms:
                    if synonym in dal:
                        activeness = activeness + dal[synonym][0]
                        pleasantness = pleasantness + dal[synonym][1]
                        imagery = imagery + dal[synonym][2]
                        synonym_found = 1
                        count = count + 1
                        
                if synonym_found == 0:
                    for antonym in antonyms:
                        if antonym in dal:
                            activeness = activeness + (-1) * dal[antonym][0]
                            pleasantness = pleasantness + (-1) * dal[antonym][1]
                            imagery = imagery + (-1) * dal[antonym][2]
                            count = count + 1
            except:
                # do nothing
                pass
        # comment till here to get output till part 5
    
    if(count == 0):
        count = 1
    
    return (activeness/count, pleasantness/count, imagery/count)
        

def get_wordnet_pos(tag):
    '''
    This function converts nltk pos tag to wordnet pos tag for a synset
    '''
    tag = tag[0].upper()
    
    tag_dict = {"J": "a", # adj
                "N": "n", # noun
                "V": "v", # verb
                "R": "r"} # adverb
    
    return tag_dict.get(tag, "n")


def main():
    list_of_tuples = load_corpus("train.txt")

    dal_dict = load_dal("dict_of_affect.txt")
    
    len_list_of_tuples = len(list_of_tuples)
    
    counter = 0
    
    for i in range(len_list_of_tuples):    
        tokenized_snippet = tokenize(list_of_tuples[i][0])
        tokenized_snippet_with_edits = tag_edits(tokenized_snippet)
        tokenized_snippet_with_edits_and_neg = tag_negation(tokenized_snippet_with_edits)
    
        list_of_tuples[i] = list(list_of_tuples[i])
        list_of_tuples[i][0] = tuple(tokenized_snippet_with_edits_and_neg)
        list_of_tuples[i] = tuple(list_of_tuples[i])
        
        for j in range(len(list_of_tuples[i][0])):
            if list_of_tuples[i][0][j][0] not in feature_dictionary and "EDIT_" not in list_of_tuples[i][0][j][0]:
                feature_dictionary[list_of_tuples[i][0][j][0]] = counter
                counter = counter + 1
    
    if DAL_FLAG:
        for item in ['dal_activeness','dal_pleasantness','dal_imagery']:
            feature_dictionary[item] = counter
            counter = counter + 1
    
    V = len(feature_dictionary)
    
    X_train = np.empty(shape=(len_list_of_tuples, V))  
        
    Y_train = np.empty(len_list_of_tuples)
    
    for i in range(len_list_of_tuples):
        feature_vec = get_features(list_of_tuples[i][0])
        
        X_train[i] = feature_vec
        Y_train[i] = list_of_tuples[i][1]

    normalized_X_train = normalize(X_train)
    
    nb = GaussianNB()
    nb.fit(normalized_X_train, Y_train)
    
    # test data
    test_list = load_corpus("test.txt")
    for i in range(len(test_list)):
        test_tokenized_snippet = tokenize(test_list[i][0])
        test_tokenized_snippet_with_edits = tag_edits(test_tokenized_snippet)
        test_tokenized_snippet_with_edits_and_neg = tag_negation(test_tokenized_snippet_with_edits)
    
        test_list[i] = list(test_list[i])
        test_list[i][0] = tuple(test_tokenized_snippet_with_edits_and_neg)
        test_list[i] = tuple(test_list[i])
        
    X_test = np.empty(shape=(len(test_list), V))
    Y_true = np.empty(len(test_list))
    
    for i in range(len(test_list)):
        test_feature_vec = get_features(test_list[i][0])
        
        X_test[i] = test_feature_vec
        Y_true[i] = test_list[i][1]

    normalized_X_test = normalize(X_test)
    
    Y_pred = nb.predict(normalized_X_test)
    
    (precision, recall, fmeasure) = evaluate_predictions(Y_pred, Y_true)
    
    print "Results for NB", (precision, recall, fmeasure)
    
    # Logistic regression
    lr = LogisticRegression()
    lr.fit(normalized_X_train, Y_train)
    
    Y_pred_lr = lr.predict(normalized_X_test)
    
    (precision_lr, recall_lr, fmeasure_lr) = evaluate_predictions(Y_pred_lr, Y_true)
    
    print "Results for LR:", (precision_lr, recall_lr, fmeasure_lr)
    
    top_features(lr, 10)
    
    out1 = get_features((('i','-'),('i','-'),('the','-')))
    out2 = get_features((('i','-'),('EDIT_am','-'),('the','-')))
    
    print out1[71], out1[0], sum(out1)
    print out2[71], out2[0], out2[3079], sum(out2)
    
    print feature_dictionary['i']
    print feature_dictionary['the']
    print feature_dictionary['am']
    
        

if __name__ == "__main__":
    main()             
        



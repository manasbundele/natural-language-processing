# @author: Manas Bundele
import math
import random
import numpy as np
random.seed(1)

############################################################################
#  Utility Functions
############################################################################

def load_file_into_list(corpus_path):
    ''' this function loads file into list and return a list of lines '''
    text_file = open(corpus_path, "r")
    list_of_lines = text_file.read().splitlines()
    
    # removing empty strings
    list_of_lines = [line for line in list_of_lines if line]
    
    text_file.close()
    
    return list_of_lines


def left_padding(n, text):
    ''' padding the left side of text with n-1 '<s>' for n gram '''
    # text refers to a string
    padding = '<s> ' * (n-1)
    return padding + text


def right_padding(text):
    ''' padding the right side of text '</s>' for n gram '''
    # text refers to a string
    return text + ' </s>'


def pad_corpus(n, corpus):
    ''' this function pads each line of the corpus on left and right side '''
    # input is list of lines
    for i in range(len(corpus)):
        t1 = left_padding(n, corpus[i])
        t2 = right_padding(t1)
        corpus[i] = t2
        
    # returns list of padded sentences
    return corpus
        
    
def tokenize(line):
    # splitting line on the basis of whitespace
    text_tokens = [words for words in line.split(" ")]
    
    # removing empty strings
    text_tokens = [token for token in text_tokens if token]
    
    return text_tokens
    
    
def convert_corpus_to_list_of_tokens(corpus):
    ''' this function converts corpus to a list of tokens '''
    # input is padded_list_of_lines, converts to word tokens
    text_tokens = [words for lines in corpus for words in lines.split(" ")]
    
    # removing empty strings
    text_tokens = [token for token in text_tokens if token]
    
    return text_tokens

    
def get_ngrams(n, text):
    ''' n gram generator function, text is tuple of strings '''
    # ngram tuple
    ngram_tup = ()

    word_context=[]
    for i in range(len(text)):
        text_tokens = tokenize(text[i])
        word_context += [(text_tokens[i+n-1],tuple(text_tokens[i:i+n-1])) for i in range(len(text_tokens)-n+1)]
        
    # dont put this in upper loop, takes lots of time to generate ngrams
    ngram_tup += tuple(word_context)    
    
    return ngram_tup
       

def create_ngramlm(n, corpus_path, mask_the_corpus=0):
    ''' This function creates ngram language model without interpolation,
        takes a parameter to see if we want to mask the corpus
    '''
    
    list_of_lines = load_file_into_list(corpus_path)
    
    padded_list_of_lines = pad_corpus(n, list_of_lines)
    
    if(mask_the_corpus == 1):
        padded_list_of_lines = mask_rare(padded_list_of_lines)
        
    ngram_model = NGramLM(n)
    ngram_model.update(padded_list_of_lines)
    
    return ngram_model
    
 
def text_prob(model, text, delta=0, katz_backoff=0, beta=0):
    ''' This function predicts log probability for a list of sentences given delta, beta and katz backoff flag
    
    '''
    # splitting text on the basis of whitespace if given a string
    if type(text) is str:
        text = [text]
    
    n = model.n
    text_with_padding = pad_corpus(n, text)
    
    log_prob = 0.0
    
    # if we dont want to use katz backoff
    if (katz_backoff == 0):    
        for j in range(len(text_with_padding)):
            text_tokens = tokenize(text_with_padding[j])
            for i in range(len(text_tokens)-n+1):
                log_prob += math.log(model.word_prob(text_tokens[i+n-1], tuple(text_tokens[i:i+n-1]), delta))
    else:
        for j in range(len(text_with_padding)):
            text_tokens = tokenize(text_with_padding[j])
            for i in range(len(text_tokens)-n+1):
                log_prob += math.log(model.katz_backoff(text_tokens[i+n-1], tuple(text_tokens[i:i+n-1]), beta))
        
    print log_prob
    return log_prob
        

def find_word_count(text_tokens):
    ''' This function finds the count of a word in the text_tokens and stores in a dictionary
    '''
    word_count = {}
    
    for i in range(len(text_tokens)):
        # fnding word count
        if(text_tokens[i] in word_count):
            word_count[text_tokens[i]] = word_count[text_tokens[i]] + 1
        else:
            word_count[text_tokens[i]] = 1 
        
    return word_count
    

def mask_rare(corpus):
    ''' This function masks the words occuring only once in the corpus(padded_list_of_lines) with <unk> token 
    '''
    
    text_tokens = convert_corpus_to_list_of_tokens(corpus)
    
    word_count = find_word_count(text_tokens)
            
    for i in range(len(corpus)):
        line = corpus[i]
        tokens = line.split(" ")
        for j in range(len(tokens)):
            if(tokens[j] in word_count and word_count[tokens[j]] == 1):
                tokens[j] = '<unk>'
        
        corpus[i] = ' '.join(tokens)
        
    return corpus


def find_token_count(text_tokens):
    ''' This function finds the total number of tokens in a text_tokens
    '''
    return len(text_tokens)



def random_text(model, max_length, delta=0):
    ''' this function generates random sentence using random_word function to generate word based on context '''
    n = model.n
    
    context = ('<s>',) * (n-1)
    sentence = (context)
    
    for i in range(max_length):
        word = model.random_word(context, delta=0)
        
        sentence += (word,)
        
        if(word == '</s>'):
            break
        else:
            context = tuple(sentence[i:i+n-1])

    print(' '.join(sentence))



def likeliest_text(model, max_length, delta=0):
    ''' this function generates random sentence using likeliest_word function to generate word based on context '''
    n = model.n
    
    context = ('<s>',) * (n-1)
    
    sentence = (context)
    
    for i in range(max_length):
        word = model.likeliest_word(context, delta=0)
        
        sentence += (word,)
        if(word == '</s>'):
            break
        else:
            context = tuple(sentence[i+1:i+n])
            
    print(' '.join(sentence))


############################################################################
#  Basic N-Gram Language Model
############################################################################
    
class NGramLM:
    def __init__(self, n):
        ''' initializing internal variables '''
        self.n = n
        self.ngram_counts = {}
        self.context_counts = {}
        self.vocabulary = set()
        self.total_contexts = 0
        # parameters for katz backoff
        self.katz_ngram_counts = {}
        self.katz_context_counts = {}
        self.word_count = {}
        self.total_words = 0
        self.alpha = {}
        
    def update(self, text):
        ''' update internal variables '''
        # getting ngrams
        ngrams = get_ngrams(self.n, text)
        
        # temporary variable initialization
        ngram_count = {}
        context_count = {}
        vocab = set(['<s>','</s>'])
        
        for i in range(len(ngrams)):
            # updating ngram counts in dictionary
            if(ngrams[i] in ngram_count):
                ngram_count[ngrams[i]] = ngram_count[ngrams[i]] + 1
            else:
                ngram_count[ngrams[i]] = 1                                
            
            # updating context counts in dictionary, ngrams[i]=(word, context)
            if(ngrams[i][1] in context_count):
                context_count[ngrams[i][1]] = context_count[ngrams[i][1]] + 1
            else:
                context_count[ngrams[i][1]] = 1                                
        
            # updating vocabulary in dictionary, ngrams[i]=(word, context)
            if (ngrams[i][0] not in vocab):
                vocab.add(ngrams[i][0])
                    
        # count total number of context for laplace smoothing
        self.total_contexts = len(context_count.keys())
        
        self.ngram_counts = ngram_count
        self.context_counts = context_count
        self.vocabulary = vocab   
        
        word_count = {}
            
        text_tokens = [token for line in text for token in tokenize(line)]
        
        token_count = len(text_tokens)
        for i in range(token_count):
            if(text_tokens[i] in word_count):
                word_count[text_tokens[i]] = word_count[text_tokens[i]] + 1
            else:
                word_count[text_tokens[i]] = 1
            
        
        self.word_count = word_count
        self.total_words = token_count


    def word_prob(self, word, context, delta=0):
        ''' calculating probability of an ngram for a given context '''
        
        vocab =  self.vocabulary
        vocab_len = len(vocab)
        
        if(word not in vocab):
            word='<unk>'
        
        context = list(context)
        
        for i in range(len(context)):
            if(context[i] not in vocab):
                context[i] = '<unk>'
                
        context = tuple(context)
        
        if (context not in self.context_counts):
            # if context not seen, assign probability as 1/(size of vocab)
            prob = 1.0/vocab_len
        else:
            if((word, context) in self.ngram_counts):
                prob = float(self.ngram_counts[(word, context)] + delta)/(self.context_counts[context] + float(delta * self.total_contexts))
            else:
                prob = float(delta)/(self.context_counts[context] + float(delta * self.total_contexts))
        
        return prob
    
    
    def find_katz_backoff_ngrams(self, text, beta):
        ''' this function stores ngrams and other internal variables for katz backoff,
            similar to update function for ngram model
        '''
        
        for i in range(self.n):
            # storing all of n,n-1,....1-gram model details in ngram_counts, context_counts
            ngrams = get_ngrams(self.n-i, text)
        
            # temporary variable initialization
            ngram_count = {}
            context_count = {}
            
            for i in range(len(ngrams)):
                # updating ngram counts in dictionary
                if(ngrams[i] in ngram_count):
                    ngram_count[ngrams[i]] = ngram_count[ngrams[i]] + 1
                else:
                    ngram_count[ngrams[i]] = 1                                
                
                # updating context counts in dictionary, ngrams[i]=(word, context)
                if(ngrams[i][1] in context_count):
                    context_count[ngrams[i][1]] = context_count[ngrams[i][1]] + 1
                else:
                    context_count[ngrams[i][1]] = 1                                
            
            self.katz_ngram_counts.update(ngram_count)
            self.katz_context_counts.update(context_count)    
            
        # subtracting beta from all ngram values
        self.katz_ngram_counts = {k:(float(v)-beta) for k,v in self.katz_ngram_counts.items()}  
        
        alpha = self.alpha
        # calculating alpha[context] values for all contexts in all ngrams
        for k,v in self.katz_ngram_counts.items():
            if(k[1] in alpha):
                alpha[k[1]] += -1 * (float(v)/self.katz_context_counts[k[1]])
            else:
                alpha[k[1]] = 1.0
            
        self.alpha = alpha
        
          
    
    def katz_backoff(self, word, context, beta):
        ''' this is a recursive function that calculates probability using katz backoff formula '''
        katz_ngram_counts = self.katz_ngram_counts
        katz_context_counts = self.katz_context_counts
        
        if(word not in self.vocabulary):
            word='<unk>'
            
        context = list(context)
        
        for i in range(len(context)):
            if(context[i] not in self.vocabulary):
                context[i] = '<unk>'
                
        context = tuple(context)
        
        # if context is empty return probability of word, i.e. p_mle unigram
        if(context == ()):
            prob_discount = (float(self.word_count[word])/self.total_words)
            return prob_discount
        
        if((word, context) in katz_ngram_counts):
            # if w belongs to A(v) where A(v) = {w|c(v,w) > 0}
            prob_discount = katz_ngram_counts[(word, context)]/float(katz_context_counts[context])
        else:
            # if w belongs to B(v) where A(v) = {w|c(v,w) = 0}
            # calculating missing probabilty mass
            if(context in self.alpha):
                prob_discount = self.alpha[context] * self.katz_backoff(word, context[1:], beta)
            else:
                # if context not in alpha, then missing probability mass = 1
                prob_discount = self.katz_backoff(word, context[1:], beta)
                
    
        return prob_discount
        
        
    def random_word(self, context, delta=0):
        ''' sample a word randomly from the model's probability distribution for a context '''
        words = []
        word_probabilities = []
        count = 0
        vocab = self.vocabulary
        # step 1
        vocab = sorted(vocab)
        
        # step 2
        r = random.random()
        
        # step 3
        for i in range(len(vocab)):
            p = self.word_prob(vocab[i], context)
            if p != 0.0:
                words.append(vocab[i])
                word_probabilities.append(p)
                count = count + 1
                
        # calculating cumulative distribution function to estimate where r lies
        cdf = np.cumsum(word_probabilities)
        
        for i in range(count):
            if(r < cdf[i]):
                return words[i]
            
    

    def likeliest_word(self, context, delta=0):
        ''' sample a most likely word (high probability) from the model's probability distribution for a context '''
        words = []
        word_probabilities = []
        vocab = list(self.vocabulary)
        
        for i in range(len(vocab)):
            # calculate probability for every word in vocab
            p = self.word_prob(vocab[i], context)
            if p != 0.0:
                words.append(vocab[i])
                word_probabilities.append(p)
        
        # find index of word with max probability
        max_probword_index = word_probabilities.index(max(word_probabilities))
        
        return words[max_probword_index]
        
        
                
        
    
############################################################################
#  Linearly Interpolated N-Gram Language Model
############################################################################
        
class NGramInterpolator:
    def __init__(self, n, lambdas):
        ''' initializing internal variables '''
        self.n = n
        self.lambdas = lambdas
        self.ngram_models = [None] * n
        for i in range(n):
            # initializing n different models of n,n-1..1 grams
            self.ngram_models[i] = NGramLM(n-i)
        
    def update(self, text):
        ''' update internal variables '''
        ng_models = self.ngram_models
        for i in range(len(ng_models)):
            # updating each model
            ng_models[i].update(text) 
        
        self.ngram_models = ng_models
        
    def word_prob(self, word, context, delta=0):
        ''' calculating probability of an ngram, given a context '''
        prob = 0.0
        ng_models = self.ngram_models
        for i in range(self.n):
            # calculating for each model and using lambdas to calculate final probability
            prob += ng_models[i].word_prob(word, context[i:], delta) * self.lambdas[i]
            
        return prob
            
    
        
############################################################################
#  Perplexity
############################################################################
def perplexity(model, corpus_path, delta=0, katz_backoff=0, beta=0):
    list_of_lines = load_file_into_list(corpus_path)
    
    test_tokens = convert_corpus_to_list_of_tokens(list_of_lines)
    
    test_token_count = find_token_count(test_tokens)
    
    # text prob already pads tha corpus, so passing list_of_lines
    avg_log_prob = text_prob(model, list_of_lines, delta, katz_backoff, beta)/float(test_token_count)
    
    return math.exp(-1 * avg_log_prob)



def main():
    # Part 1.3
    print("Part 1.3:\n")
    unmasked_warpeace_trigram_model = create_ngramlm(3, "/Users/manasbundele/Documents/nlp/assignment1/warpeace.txt", mask_the_corpus=0)
    
    print("Part 1: Sentence 1 log probability: ")
    text_prob(unmasked_warpeace_trigram_model, "God has given it to me, let him who touches it beware!")

    print("Part 1: Sentence 2 log probability: Error!")
    #text_prob(unmasked_warpeace_trigram_model, "Where is the prince, my Dauphin?")
    
    print("\n\n")

    # Part 2.1
    print("Part 2.1:\n")
    masked_warpeace_trigram_model = create_ngramlm(3, "/Users/manasbundele/Documents/nlp/assignment1/warpeace.txt", mask_the_corpus=1)
    
    print("Part 2.1: Sentence 1 log probability: ")
    text_prob(masked_warpeace_trigram_model, "God has given it to me, let him who touches it beware!")

    print("Part 2.1: Sentence 2 log probability: Math Error!")
    #text_prob(masked_warpeace_trigram_model, "Where is the prince, my Dauphin?")
    print("\n\n")

    # Part 2.3
    print("Part 2.3:\n")
    print("Part 2.3: delta = 0.5, Sentence 1 log probability: ")
    text_prob(masked_warpeace_trigram_model, "God has given it to me, let him who touches it beware!", delta=0.5)

    print("Part 2.3: delta = 0.5, Sentence 2 log probability: ")
    text_prob(masked_warpeace_trigram_model, "Where is the prince, my Dauphin?", delta = 0.5)

    print("Part 2.3: delta = 0.75, Sentence 1 log probability: ")
    text_prob(masked_warpeace_trigram_model, "God has given it to me, let him who touches it beware!", delta=0.75)

    print("Part 2.3: delta = 0.75, Sentence 2 log probability: ")
    text_prob(masked_warpeace_trigram_model, "Where is the prince, my Dauphin?", delta = 0.75)

    print("Part 2.3: delta = 1, Sentence 1 log probability: ")
    text_prob(masked_warpeace_trigram_model, "God has given it to me, let him who touches it beware!", delta=1)

    print("Part 2.3: delta = 1, Sentence 2 log probability: ")
    text_prob(masked_warpeace_trigram_model, "Where is the prince, my Dauphin?", delta = 1)

    print("Part 2.3: delta = 2, Sentence 1 log probability: ")
    text_prob(masked_warpeace_trigram_model, "God has given it to me, let him who touches it beware!", delta=2)

    print("Part 2.3: delta = 2, Sentence 2 log probability: ")
    text_prob(masked_warpeace_trigram_model, "Where is the prince, my Dauphin?", delta = 2)
    print("\n\n")
    
    # Part 2.4
    print("Part 2.4:\n")
    masked_warpeace_trigram_interpolator = NGramInterpolator(3, [0.33,0.33,0.33])
    warpeace_list_of_lines = load_file_into_list("/Users/manasbundele/Documents/nlp/assignment1/warpeace.txt")
    
    warpeace_list_of_lines = pad_corpus(3, warpeace_list_of_lines)

    warpeace_list_of_lines = mask_rare(warpeace_list_of_lines)
    
    masked_warpeace_trigram_interpolator.update(warpeace_list_of_lines)
    
    print("Part 2.4: Trigram Interpolator, delta = 0, Sentence 1 log probability: ")
    text_prob(masked_warpeace_trigram_interpolator, "God has given it to me, let him who touches it beware!",0)
    
    print("Part 2.4: Trigram Interpolator, delta = 0, Sentence 2 log probability: ")
    text_prob(masked_warpeace_trigram_interpolator, "Where is the prince, my Dauphin?",0)
    
    print("Part 2.4: Trigram Interpolator, delta = 1, Sentence 1 log probability: ")
    text_prob(masked_warpeace_trigram_interpolator, "God has given it to me, let him who touches it beware!",1)
    
    print("Part 2.4: Trigram Interpolator, delta = 1, Sentence 2 log probability: ")
    text_prob(masked_warpeace_trigram_interpolator, "Where is the prince, my Dauphin?",1)
    print("\n\n")

    # Part 3.1
    print("Part 3.1:\n")
    masked_shakespeare_trigram_model = create_ngramlm(3, "/Users/manasbundele/Documents/nlp/assignment1/shakespeare.txt", mask_the_corpus=1)
    
    #perplexity_delta0 = perplexity(masked_shakespeare_trigram_model, "/Users/manasbundele/Documents/nlp/assignment1/sonnets.txt")
    print "Perplexity of test data sonnets.txt trained on shakespeare.txt without smoothing =  Math Error!"
    
    # it prints log probability value before printing perplexity
    perplexity_delta_point5 = perplexity(masked_shakespeare_trigram_model, "/Users/manasbundele/Documents/nlp/assignment1/sonnets.txt", delta=0.5)
    print "Perplexity of test data sonnets.txt trained on shakespeare.txt with smoothing 0.5 = ", perplexity_delta_point5
    print("\n\n")
    
    # Part 3.2
    print("Part 3.2:\n")
    # it prints log probability value before printing perplexity
    perplexity_warpeace_sonnets = perplexity(masked_warpeace_trigram_model, "/Users/manasbundele/Documents/nlp/assignment1/sonnets.txt", delta=0.5)
    print "Perplexity of test data sonnets.txt trained on warpeace.txt with smoothing 0.5 =  ", perplexity_warpeace_sonnets
    print("\n\n")
    
    # Part 4.1
    print("Part 4.1:\n")
    for i in range(5):
        print "Randomly generated sentence ", i+1, " : "
        random_text(masked_shakespeare_trigram_model, 10, delta=0)
    
    print("\n\n")
    
    # Part 4.2
    print("Part 4.2:\n")
    masked_shakespeare_bigram_model = create_ngramlm(2, "/Users/manasbundele/Documents/nlp/assignment1/shakespeare.txt", mask_the_corpus=1)
    
    masked_shakespeare_fourgram_model = create_ngramlm(4, "/Users/manasbundele/Documents/nlp/assignment1/shakespeare.txt", mask_the_corpus=1)
    
    masked_shakespeare_fivegram_model = create_ngramlm(5, "/Users/manasbundele/Documents/nlp/assignment1/shakespeare.txt", mask_the_corpus=1)
    
    print("Likeliest sentence for bigram:")
    likeliest_text(masked_shakespeare_bigram_model, 10, delta=0)

    print("Likeliest sentence for trigram:")    
    likeliest_text(masked_shakespeare_trigram_model, 10, delta=0)
    
    print("Likeliest sentence for 4-gram:")
    likeliest_text(masked_shakespeare_fourgram_model, 10, delta=0)
    
    print("Likeliest sentence for 5-gram:")
    likeliest_text(masked_shakespeare_fivegram_model, 10, delta=0)
    print("\n\n")
    
    # Part 6.1a
    print("Part 6.1a:\n")
    shakespeare_list_of_lines = load_file_into_list("/Users/manasbundele/Documents/nlp/assignment1/shakespeare.txt")
    
    padded_shakespeare_list_of_lines = pad_corpus(3, shakespeare_list_of_lines)
    padded_shakespeare_list_of_lines = mask_rare(padded_shakespeare_list_of_lines)    

    masked_shakespeare_trigram_model.find_katz_backoff_ngrams(padded_shakespeare_list_of_lines, 0.75)
    
    # it prints log probability value before printing perplexity
    perplexity_delta0_part6 = perplexity(masked_shakespeare_trigram_model, "/Users/manasbundele/Documents/nlp/assignment1/sonnets.txt", katz_backoff=1, beta=0.75)
    print "Perplexity of test data sonnets.txt trained on shakespeare.txt without smoothing using Katz backoff =  ", perplexity_delta0_part6
    
    # it prints log probability value before printing perplexity
    perplexity_delta_point5_part6_warpeace = perplexity(masked_warpeace_trigram_model, "/Users/manasbundele/Documents/nlp/assignment1/sonnets.txt", katz_backoff=1, beta=0.75)
    print "Perplexity of test data sonnets.txt trained on warpeace.txt using Katz backoff =  ",perplexity_delta_point5_part6_warpeace
    
    
    print("\n\n")
    
    
    
if __name__ == "__main__":
    main()
    
    
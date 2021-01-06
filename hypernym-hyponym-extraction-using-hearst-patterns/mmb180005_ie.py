import nltk
import re
import sys


# Fill in the pattern (see Part 2 instructions)
NP_grammar = 'NP: {<DT>?<JJ.*>*<NN.*>+}' 


EXTRA_CREDIT = 0

# Fill in the other 4 rules (see Part 3 instructions)
hearst_patterns = [
    ('(NP_\w+ ?(, NP_\w+)* ?(, )?(and |or )?other NP_\w+)', 'after'),
    ('(NP_\w+ (, )?such as (NP_\w+ ? (, )?(and |or )?)+)', 'before'),
    ('(such NP_\w+ as (NP_\w+ ? (, )?(and |or )?)+)', 'before'), # or (such NP_\w+ as (NP_\w+ ?(, )?(and |or )?)+)
    ('(NP_\w+ ?(, )?include (NP_\w+ ? (, )?(and |or )?)+)', 'before'),
    ('(NP_\w+ ?(, )?especially (NP_\w+ ? (, )?(and |or )?)+)', 'before'),
    
    # extra credit patterns
    ('(NP_\w+ (, )?which be call (NP_("_)?\w+(_")? ? (, )?(and |or )?)+)','before'), # X which is called Y -> 'which be call'  after lemmatization
    ('(NP_\w+ (, )?like (NP_("_)?\w+(_")? ? (, )?(and |or )?)+)', 'before'), # X like Y
    ('(NP_\w+ (\( )?e. g. (, )?(NP_("_)?\w+(_")? ? (, )?(and |or )?(\))?)+)','before'), # X (? e. g. Y )?
    ('(NP_\w+ (, )?be a (NP_("_)?\w+(_")? ? (, )?(and |or )?)+)', 'after') # X is a Y -> 'be a' after lemmatization
    ]


count_hearst = dict()

# Fill in the function (see Part 1 instructions)
# Argument type: path - string
# Return type: list of tuples
def load_corpus(path):
    
    list_of_tuples = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                line = line.split("\t")
                sentence = [word.strip() for word in line[0].split() if word]
                lemmatized = [word.strip() for word in line[1].split() if word]
                list_of_tuples.append((sentence,lemmatized))
        
    return list_of_tuples
            


# Fill in the function (see Part 1 instructions)
# Argument type: path - string
# Return type: tuple of sets
def load_test(path):
    true_set = set()
    false_set = set()
    
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                line = line.split("\t")
                hyponym = line[0].strip()
                hypernym = line[1].strip()
                label = line[2].strip()
                
                if label == "True":
                    true_set.add((hyponym, hypernym))
                else:
                    false_set.add((hyponym, hypernym))

    return (true_set, false_set)


# Fill in the function (see Part 2 instructions)
# Argument type: sentence, lemmatized - list of strings; parser - nltk.RegexpParser
# Return type: string
def chunk_lemmatized_sentence(sentence, lemmatized, parser):
    
    sent = nltk.pos_tag(sentence)
    lemm = []
    for i in range(len(sent)):
        lemm.append((lemmatized[i], sent[i][1]))
        
    tree_lemmatized = parser.parse(lemm)
    
    list_of_chunks = tree_to_chunks(tree_lemmatized)
    
    string_of_list_of_chunks = merge_chunks(list_of_chunks)
    
    return string_of_list_of_chunks


# Fill in the function (see Part 2 instructions)
# Helper function for chunk_lemmatized_sentence()
# Argument type: tree - nltk.Tree
# Return type: list of strings
def tree_to_chunks(tree):
    chunks = []
    
    for child in tree:
        if not isinstance(child, nltk.Tree):
            chunks.append(child[0])
        else:
            tokens = [tup[0] for tup in child] # format is (token,tag)
            combined_tokens = "_".join(tokens)
            tagged = "NP_" + combined_tokens
            chunks.append(tagged)
            
    return chunks
            
            

# Fill in the function (see Part 2 instructions)
# Helper function for chunk_lemmatized_sentence()
# Argument type: chunks - list of strings
# Return type: string
def merge_chunks(chunks):
    pieces_of_reconstructed_string = []
    
    for item in chunks:
        if len(pieces_of_reconstructed_string) != 0 and "NP_" in pieces_of_reconstructed_string[-1] and "NP_" in item:
            pieces_of_reconstructed_string[-1] = pieces_of_reconstructed_string[-1] + "_" + item.replace("NP_","")
        else:
            pieces_of_reconstructed_string.append(item)
            
    return " ".join(pieces_of_reconstructed_string)
                


# Fill in the function (see Part 4 instructions)
# Argument type: chunked_sentence - string
# Yield type: tuple
def extract_relations(chunked_sentence):
    global hearst_patterns
    if EXTRA_CREDIT == 0:
        hearst_patterns = hearst_patterns[0:5]
    
    for pattern in hearst_patterns:
        pattern_found = re.search(pattern[0], chunked_sentence)
        
        idx=hearst_patterns.index(pattern)
        
        if pattern_found:
            match = pattern_found.group(0)
            list_of_str = match.split()
            tokens_with_NP = [item for item in list_of_str if "NP_" in item]
            postprocessed_tokens_with_NP = postprocess_NPs(tokens_with_NP)
            
            if pattern[1] == "before":
                hypernym = postprocessed_tokens_with_NP[0]
                hyponym = postprocessed_tokens_with_NP[1:]
            elif pattern[1] == "after":
                hypernym = postprocessed_tokens_with_NP[-1]
                hyponym = postprocessed_tokens_with_NP[:-1]

            
            if idx not in count_hearst:
                count_hearst[idx] = 1
            else:
                count_hearst[idx] += 1

            for item in hyponym:
                yield (item, hypernym)
                
            
                

# Fill in the function (see Part 4 instructions)
# Helper function for extract_relations()
# Argument type: list of strings
# Return type: list of strings
def postprocess_NPs(NPs):
    output = []
    for item in NPs:
        np_removed = item.replace("NP_","")
        underscore_removed = np_removed.replace("_"," ")
        
        output.append(underscore_removed)
        
    return output


# Fill in the function (see Part 5 instructions)
# Argument type: extractions, gold_true, gold_false - set of tuples
# Return type: tuple
def evaluate_extractions(extractions, gold_true, gold_false):
    tp = len(extractions & gold_true)
    fp = len(extractions & gold_false)
    fn = len(gold_true - extractions)
    
    print("tp = ",tp,"fp = ", fp, "fn = ", fn)
    
    precision = float(tp)/(tp + fp)
    recall = float(tp)/(tp + fn)
    fmeasure = (2 * precision * recall)/float(precision + recall)
    
    return (precision, recall, fmeasure)


def main(args):
    corpus_path = args[0]
    test_path = args[1]

    wikipedia_corpus = load_corpus(corpus_path)    
    test_true, test_false = load_test(test_path)
    
    NP_chunker = nltk.RegexpParser(NP_grammar)

    # Complete the line (see Part 2 instructions)
    wikipedia_corpus = [chunk_lemmatized_sentence(sentence, lemmatized, NP_chunker) for (sentence,lemmatized) in wikipedia_corpus]

    extracted_pairs = set()
    for chunked_sentence in wikipedia_corpus:
        for pair in extract_relations(chunked_sentence):
            extracted_pairs.add(pair)

    print("Count of heart pattern by (index,count) format in dictionary: " % count_hearst)
    print('Precision: %f\nRecall:%f\nF-measure: %f' % evaluate_extractions(extracted_pairs, test_true, test_false))

if __name__ == '__main__':
    #sys.exit(main(sys.argv[1:]))
    main(sys.argv[1:])
    
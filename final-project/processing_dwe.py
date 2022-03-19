import pandas as pd
import ast
import dask.dataframe as dd
from gensim.utils import effective_n_jobs
import spacy
try:
    nlp = spacy.load("en")
except OSError:
    nlp = spacy.load("en_core_web_sm")

path = 'Environmental-Discourse'
env = pd.read_pickle('../Data/'+path+'/env_0.pkl')

def word_tokenize(word_list, model=nlp, MAX_LEN=1500000):
    
    tokenized = []
    if type(word_list) == list and len(word_list) == 1:
        word_list = word_list[0]

    if type(word_list) == list:
        word_list = ' '.join([str(elem) for elem in word_list]) 
    # since we're only tokenizing, I remove RAM intensive operations and increase max text size

    model.max_length = MAX_LEN
    doc = model(word_list, disable=["parser", "tagger", "ner", "lemmatizer"])
    
    for token in doc:
        if not token.is_punct and len(token.text.strip()) > 0:
            tokenized.append(token.text)
    return tokenized

def normalizeTokens(word_list, extra_stop=[], model=nlp, lemma=True, MAX_LEN=1500000):
    #We can use a generator here as we just need to iterate over it
    normalized = []
    if type(word_list) == list and len(word_list) == 1:
        word_list = word_list[0]

    if type(word_list) == list:
        word_list = ' '.join([str(elem) for elem in word_list]) 

    # since we're only normalizing, I remove RAM intensive operations and increase max text size

    model.max_length = MAX_LEN
    doc = model(word_list.lower(), disable=["parser", "ner"])

    if len(extra_stop) > 0:
        for stopword in extra_stop:
            lexeme = nlp.vocab[stopword]
            lexeme.is_stop = True

    # we check if we want lemmas or not earlier to avoid checking every time we loop
    if lemma:
        for w in doc:
            # if it's not a stop word or punctuation mark, add it to our article
            if w.text != '\n' and not w.is_stop and not w.is_punct and not w.like_num and len(w.text.strip()) > 0:
            # we add the lematized version of the word
                normalized.append(str(w.lemma_))
    else:
        for w in doc:
            # if it's not a stop word or punctuation mark, add it to our article
            if w.text != '\n' and not w.is_stop and not w.is_punct and not w.like_num and len(w.text.strip()) > 0:
            # we add the lematized version of the word
                normalized.append(str(w.text.strip()))

    return normalized

def ngram_tagger(tokens):
    n = len(tokens)
    i = 0
    tokens_q = []
    tokens_qt = []
    tokens_qtb = []
    
    # quadgrams
    while i < n:
        words = '_'.join(tokens[i:i+4])
        if words in quadgrams:
            tokens_q.append(words)
            i += 4
        else:
            tokens_q.append(tokens[i])
            i += 1
    
    # trigrams
    n = len(tokens_q)
    i = 0
    while i < n:
        words = '_'.join(tokens_q[i:i+3])
        if words in trigrams:
            tokens_qt.append(words)
            i += 3
        else:
            tokens_qt.append(tokens_q[i])
            i += 1
    
    # bigrams
    n = len(tokens_qt)
    i = 0
    while i < n:
        words = '_'.join(tokens_qt[i:i+2])
        if words in bigrams:
            tokens_qtb.append(words)
            i += 2
        else:
            tokens_qtb.append(tokens_qt[i])
            i += 1
    
    return tokens_qtb

def sent_tokenize(word_list, model=nlp):
    doc = model(word_list)
    sentences = [sent.text.strip() for sent in doc.sents]
    return sentences

quadgrams = [('intergovernmental', 'panel', 'climate', 'change'),
             ('natural', 'resources', 'defense', 'council'),
             ('coal', 'fired', 'power', 'plants'),
             ('national', 'oceanic', 'atmospheric', 'administration')]

tr = pd.read_csv('../Data/' + path + '/trigrams.csv', converters={'Unnamed: 0': ast.literal_eval})
tr.columns = ['trigram', 'freq', 'tag']
trigrams = [t for t in tr[tr.tag == 1].trigram]

b = pd.read_csv('../Data/' + path + '/bigrams.csv', converters={'Unnamed: 0': ast.literal_eval})
b.columns = ['bigram', 'freq', 'tag']
bigrams = [t for t in b[b.tag == 1].bigram]

quadgrams = ['_'.join(t) for t in quadgrams]
trigrams = ['_'.join(t) for t in trigrams]
bigrams = ['_'.join(t) for t in bigrams]



d_env = dd.from_pandas(env, npartitions=effective_n_jobs(-1))
d_env['sents'] = d_env.text.map(lambda x: [ngram_tagger(
                                           normalizeTokens(
                                           word_tokenize(s), lemma=False)) for s in sent_tokenize(x)])
d_env['sents'] = d_env.sents.map(lambda x: [s for s in x if len(s)>0])
env = d_env.compute()

env.to_pickle('../Data/'+path+'/env_processed_sent.pkl')
env.to_csv('../Data/'+path+'/env_processed_sent.csv')
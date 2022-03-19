# IMPORT PACKAGES
import pandas as pd
import nltk
from nltk.util import ngrams
import sklearn
from sklearn.model_selection import train_test_split
import ast
import dask.dataframe as dd
from dask.multiprocessing import get
import spacy
import gensim
from gensim import corpora, models
from gensim.utils import effective_n_jobs
from dask.distributed import Client
from dask import delayed
import multiprocessing as mp
import dask
try:
    nlp = spacy.load("en")
except OSError:
    nlp = spacy.load("en_core_web_sm")
    
# CREATE FUNCTIONS
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

def dropMissing(wordLst, vocab):
    return [w for w in wordLst if w in vocab]

def tm(dictionary, corpus, n_topics):
    model = models.ldamodel.LdaModel(corpus=corpus,
                                     id2word=dictionary,
                                     num_topics=n_topics,
                                     alpha=0.1, eta=0.01)
    coherence_model = models.coherencemodel.CoherenceModel(model=model, 
                                                           corpus=corpus,
                                                           dictionary=dictionary,
                                                           coherence='u_mass')
    
    coherence = coherence_model.get_coherence()
    
    topicsDict = {}
    for topicNum in range(n_topics):
        topicWords = [w for w, p in model.show_topic(topicNum)]
        topicsDict['Topic_{}'.format(topicNum)] = topicWords

    wordRanksDF = pd.DataFrame(topicsDict)
    
    return model, coherence, wordRanksDF


path = 'Environmental-Discourse'

print('Reading in data...')
env = pd.read_pickle('../Data/' + path + '/env_0.pkl')

print('Creating n-gram lists...')
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

print('Tokeninzing...')
d_env = dd.from_pandas(env, npartitions=effective_n_jobs(-1))
d_env['tokens_full'] = d_env.text.map(lambda x: ngram_tagger(normalizeTokens(word_tokenize(x))), meta=('x', str))
d_env['text_reconstructed'] = d_env.tokens_full.map(lambda x: ' '.join(x))
env_tok = d_env.compute()

print('Reducing data by TF vocabulary...')
TFIDFVectorizer = sklearn.feature_extraction.text.TfidfVectorizer(max_df=0.5, 
                                                                  max_features=10000, 
                                                                  min_df=3, 
                                                                  norm='l2')
TFIDFVects = TFIDFVectorizer.fit_transform(env_tok.text_reconstructed)

d_env = dd.from_pandas(env_tok, npartitions=effective_n_jobs(-1))
d_env['tokens_reduced'] = d_env.tokens_full.map(lambda x: dropMissing(x, TFIDFVectorizer.vocabulary_.keys()))
env_tok = d_env.compute()


print('Creating dictionary, bow corpus, tfidf...')
dictionary = corpora.Dictionary([i for i in env_tok.tokens_reduced])
bow_corpus = [dictionary.doc2bow(text) for text in env_tok.tokens_reduced]

print('Running model...')
model, coherence, top_words = tm(dictionary, bow_corpus, 100)

print('Saving models, top words, and coherence scores...')
model.save('../Data/' + path + '/Full-TMs/Models/lda_100')

print('Append topic-document loadings to dataframe...')
env_tok['lda_topics'] = [model[dictionary.doc2bow(l)] for l in env_tok['tokens_reduced']]

#Dict to temporally hold the probabilities
topicsProbDict = {i : [0] * len(env_tok) for i in range(model.num_topics)}

#Load them into the dict
for index, topicTuples in enumerate(env_tok['lda_topics']):
    for topicNum, prob in topicTuples:
        topicsProbDict[topicNum][index] = prob

#Update the DataFrame
col_names = []
for topicNum in range(model.num_topics):
    env_tok['topic_{}'.format(topicNum)] = topicsProbDict[topicNum]
    col_names.append('topic_{}'.format(topicNum))

env_doc_topic = env_tok[['source','url','title','date','text','year'] + col_names]
env_doc_topic.to_csv('../Data/'+path+'/Full-TMs/lda100_loadings.csv')
env_doc_topic.to_pickle('../Data/'+path+'/Full-TMs/lda100_loadings.pkl')

print("COMPLETE!")

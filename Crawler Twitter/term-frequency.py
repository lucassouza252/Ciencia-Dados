import operator 
import json
import processing
import graph
from collections import Counter
from nltk.corpus import stopwords
import string
from nltk import bigrams 

dicionario = ['rt', 'via', 'RT', '…', ';', 'a', 'ou', "",
            'aff', ',', ':', 'é', 'ção', 'ça', 'A', 'ão', 'O', 'popula', 'est', 'come', 'ás',
            'vão', 'pré', 'pre', 'pol', 'ções', 'á', 'ços', 'ço', 'q', 'sá', 'd', 'E', 'e', 'pra',
            'ém', '🇷', '🇧', 'não', 'ser', 'Não', '30', 'boa', 'ência', 'vai']

fname = 'cpidacovid.json'

with open(fname, 'r') as f:
    count_all = Counter()
    for line in f:
        tweet = json.loads(line)
        terms_all = [term for term in processing.preprocess(tweet['text'])]


        punctuation = list(string.punctuation)
        stop = stopwords.words('portuguese') + punctuation + dicionario

        terms_stop = [term for term in terms_all if term not in stop]

        # Count terms only once, equivalent to Document Frequency
        terms_single = set(terms_all)

        # Count hashtags only
        terms_hash = [term for term in processing.preprocess(tweet['text']) 
                    if term.startswith('#')]

        # Count terms only (no hashtags, no mentions)
        terms_only = [term for term in terms_stop if term not in stop and 
                        not term.startswith(('#', '@'))] 
                    # mind the ((double brackets))
                    # startswith() takes a tuple (not a list) if 
                    # we pass a list of inputs

        
        count_all.update(terms_only)
    
    common = (count_all.most_common(20))

    #common.remove(('️', 24))

    cinco_maior = graph.cinco_maior(common)    
    lista_cinco = graph.lista_cinco_maior(common, cinco_maior)
    print(common)
    graph.plotgraph(common, lista_cinco)
import numpy as np
import pymorphy2
import gensim
import re
from nltk.corpus import stopwords

m = pymorphy2.MorphAnalyzer()

pymorphy2_dict = {
    'NOUN': 'NOUN', 
    'ADJF': 'ADJ', 
    'ADJS': 'ADJ',
    'COMP': 'ADJ', 
    'VERB': 'VERB',
    'INFN': 'VERB', 
    'PRTF': 'ADV',                                                                                                                                                                                                                                                                    
    'PRTS': 'ADV',
    'GRND': 'ADV',
    'NUMR': 'NUM',                                                                                                                                                                                                                                                                    
    'NPRO': 'PRON',                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
    'CONJ': 'CONJ',                                                                                                                                                                                                                                                                  
    'INTJ': 'INTJ',                                                                                                                                                                                                 
    'PRCL': 'PART',  
    'PREP': 'ADP',                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
    'SPRO': 'PRON',
    'ADVB': 'ADV',
    'PRED': 'ADV'
 
}


def tag_word(word, with_tags=False, only_tags=False):
    processed = m.parse(word)

    if with_tags:
        try:
            lemma = processed[0].normal_form + "_" + pymorphy2_dict[processed[0].tag.POS]
        except KeyError:
            print(processed[0])
            lemma = processed[0].normal_form + "_X"
    elif only_tags:
        try:
            lemma = word +  "_" + pymorphy2_dict[processed[0].tag.POS]
        except KeyError:
            lemma = word + "_X"
    else:
        lemma = processed[0].normal_form 
    return lemma


def tokenizer(query, with_tags=False, only_tags=False):
    query = query.lower()
    query = re.sub(r'[^\w\s]',' ',query)
    query = re.sub(r'\d+','',query)
    query = re.sub(r'\s+',' ',query)
    query = query.strip()
    updated_query = []
    for word in query.split(' '):
        if len(word) > 3:
            word = tag_word(word, with_tags, only_tags)
            if word not in stopwords.words("russian"):
                updated_query.append(word)
    return updated_query

def tokenizer_only_tags(query):
    return tokenizer(query,  with_tags=False,  only_tags=True)

def tokenizer_lemma_tags(query):
    return tokenizer(query, with_tags=True, only_tags=False)
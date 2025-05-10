import os, json, errno, requests
import numpy as np
from copy import deepcopy
from operator import itemgetter 
import torch 
import torch.optim as optim 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset 

from data_generation.nlp import ALICE_URL, WIZARD_URL 
from stepbystep.v4 import StepByStep 
from seq2seq import * 

import nltk 
from nltk.tokenize import sent_tokenize

import gensim 
from gensim import corpora, downloader
from gensim.parsing.preprocessing import *
from gensim.utils import simple_preprocess
from datasets import load_dataset, Split
from textattack.augmentation import EmbeddingAugmenter
from transformers import BertTokenizer

localfolder = 'texts'

# ## Notice: 
# ## Download data
# fname1 = os.path.join(localfolder, 'alice28-1476.txt')
# with open(fname1, 'r') as f:
#     alice = ''.join(f.readlines()[104:3704])
# fname2 = os.path.join(localfolder, 'wizoz10-1740.txt')
# with open(fname2, 'r') as f:
#     wizard = ''.join(f.readlines()[310:5100])

# print(alice[:500])
# print('\n')
# print(wizard[:500])

# text_cfg = """fname,start,end
# alice28-1476.txt,104,3704
# wizoz10-1740.txt,310,5100"""
# bytes_written = open(os.path.join(localfolder, 'lines.cfg'), 'w').write(text_cfg)

# ## Sentence Tokenization
# sentence = "I'm following the white rabbit"
# tokens = sentence.split(' ')
# print("Tokens example manually: ", tokens)
# using lib
# nltk.download('punkt_tab')
# nltk.download('punkt')
# corpus_alice = sent_tokenize(alice)
# corpus_wizard = sent_tokenize(wizard)
# print("Number of sentences(corpus) 'alice' and 'wizard': ", len(corpus_alice), len(corpus_wizard))
# print("One corpus: ", corpus_wizard[30])

def sentence_tokenize(source, quote_char='\\', sep_char=',', include_header=True,
    include_source=True, extensions=('txt'), **kwargs):
    nltk.download('punkt')
    # if source is a folder, get a list of .txt file names
    if os.path.isdir(source):
        # splitext("test.txt")->["test",".txt"]
        filenames = [
            f for f in os.listdir(source)
            if os.path.isfile(os.path.join(source, f))
            and os.path.splitext(f)[1][1:] in extensions
        ]
    elif isinstance(source, str):
        filenames = [source]
    
    # if there is a .cfg file, build a dict with corresponding start and end lines of each text file
    # {"a.txt": (200,302), "b.txt": (130, 900)}
    config_file = os.path.join(source, 'lines.cfg')
    config = {}
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            rows = f.readlines()
        for r in rows[1:]:
            fname, start, end = r.strip().split(',')
            config.update({fname:(int(start), int(end))})

    new_fnames = []
    for fname in filenames:
        # if we wanna cut out part of a file
        try:
            start, end = config[fname]
        except KeyError:
            start, end = None, None 

        # remove newline \n to more accurately take sentences
        with open(os.path.join(source, fname), 'r') as f:
            contents = (
                ''.join(f.readlines()[slice(start,end,None)])
                .replace('\n', ' ').replace('\r', '')
            )
        corpus = sent_tokenize(contents, **kwargs)
        
        # build a csv file containing tokenized sentences
        base = os.path.splitext(fname)[0] # ["abc",".txt"]
        new_fname = f'{base}.sent.csv'
        new_fname = os.path.join(source, new_fname)
        with open(new_fname, 'w') as f:
            if include_header:
                if include_source:
                    f.write('sentence,source\n')
                else:
                    f.write('sentence\n')
            for sentence in corpus:
                if include_source:
                    f.write(f'{quote_char}{sentence}{quote_char}{sep_char}{fname}\n')
                else:
                    f.write(f'{quote_char}{sentence}{quote_char}\n')
        new_fnames.append(new_fname)
    return sorted(new_fnames)

# new_fnames = sentence_tokenize(localfolder)
# print("New tokenized .csv files: ", new_fnames)

# ## HuggingFace
# # Loading dataset
# dataset = load_dataset(
#     path='csv',
#     data_files=new_fnames,
#     quotechar='\\',
#     split=Split.TRAIN 
# )
# print("Dataset columns: ", dataset.features) # {'sentence':'','source':''}
# print("Dataset number of columns: ", dataset.num_columns)
# print("Dataset shape: ", dataset.shape)
# print("Dataset sentence number 2: ", dataset[2]) # {'sentence':'xxx', 'source':'alice28.txt'}
# print("Dataset first 3 file-sources: ", dataset['source'][:3])
# print("Dataset source files: ", dataset.unique('source')) # ['a.txt','b.txt']
# # Add new column 'labels' to the dataset, 0/1 means belongs to 'alice' source or not
# # row: {'sentence': 'xxx', 'source':'alice.txt'}-> return {'label':0 or 1}
# def is_alice_label(row):
#     is_alice = int(row['source'] == 'alice28-1476.txt')
#     return {'labels': is_alice}
# dataset = dataset.map(is_alice_label) # {'labels':1,'sentence':'xxx','source':'alice.txt'}
# print ("Dataset (added 'labels' column):\n", dataset)
# # dataset shuffle and train test split
# shuffled_dataset = dataset.shuffle(seed=42)
# split_dataset = shuffled_dataset.train_test_split(test_size=0.2)
# # {'train':Dataset(), 'test':Dataset()}; each is {'features':['sentence','source','labels'], 'num_rows': 3081}
# print("Split dataset:\n", split_dataset) 
# train_dataset = split_dataset['train']
# test_dataset = split_dataset['test']

# ## Word tokenization
# sentence = "I'm following the white rabbit"
# print("Word tokenized:", preprocess_string(sentence)) # ['follow','white','rabbit']
# filters = [
#     lambda x: x.lower(),
#     strip_tags,
#     strip_punctuation,
#     strip_multiple_whitespaces,
#     strip_numeric 
# ]
# print("Word tokenized with filters: ", preprocess_string(sentence, filters=filters))
# tokens = simple_preprocess(sentence)
# print("Word tokenized with simple_preprocess: ", tokens)

# ## Data Augmentation
# augmenter = EmbeddingAugmenter()
# feynman = 'What I cannot create, I do not understand.'
# # Augment 4 new sentences
# print("Augmented sentences:\n")
# for i in range(4):
#     print(augmenter.augment(feynman))

# ## Vocabulary
# sentences = train_dataset['sentence']
# tokens = [simple_preprocess(sent) for sent in sentences] # [[<tokens of sentence 1>],[<tokens of sentence 2],...]
# print("1st sentence's tokens: ", tokens[0])
# dictionary = corpora.Dictionary(tokens)
# print("Dictionary: ", dictionary) # Dictionary<3699 unique tokens: [...]>
# print("Dict number of sentences: ", dictionary.num_docs)
# print("Dict number of words: ", dictionary.num_pos)
# print("Dict entries with id:\n", dictionary.token2id) # {'attends':0,'apple':1,...}
# print("Vocabulary:\n", list(dictionary.token2id.keys())) # ['attends', 'apple']
# print("Dict collection frequencies:\n", dictionary.cfs)
# print("In how many sentences does each entry appear? ", dictionary.dfs)
# # Check a list of tokens in a sentence, each belong to which index in dictionary?
# sentence = 'follow the white rabbit xcxzc'
# new_tokens = simple_preprocess(sentence)
# ids = dictionary.doc2idx(new_tokens) # [524,23,469,459]
# print(f"Tokens {new_tokens} have indices {ids}") # -1 it no exist
# # Add new tokens to dictionary
# special_tokens = {'[PAD]':0, '[UNK]':1}
# dictionary.patch_with_special_tokens(special_tokens)
# Get rare ids, to filter out rare tokens 
def get_rare_ids(dictionary, min_freq):
    rare_ids = [
        t[0] for t in dictionary.cfs.items()
        if t[1] < min_freq]
    return rare_ids 

# save vocabularies to vocab.txt
def make_vocab(sentences, folder=None, special_tokens=None, vocab_size=None, min_freq=None):
    if folder is not None:
        if not os.path.exists(folder):
            os.mkdir(folder)
    # tokenizes the sentences
    tokens = [simple_preprocess(sent) for sent in sentences] # [['ac','zx'],['a','abh','zx'],..]
    dictionary = corpora.Dictionary(tokens)
    # keeps only the most frequent <vocab_size> words
    if vocab_size is not None:
        dictionary.filter_extremes(keep_n=vocab_size)
    # remove rare words (in case the most-frequent dict still contains rare words)
    if min_freq is not None:
        rare_tokens = get_rare_ids(dictionary, min_freq)
        dictionary.filter_tokens(bad_ids=rare_tokens)
    # gets the list of tokens and frequencies 
    items = dictionary.cfs.items() # {'apple': 23,'orange': 15}
    # sort the tokens in descending order => return ['test','common',...]
    words = [
        dictionary[t[0]]
        for t in sorted(items, key=lambda t: -t[1])
    ]
    # add special tokens to the list of words, if any
    if special_tokens is not None:
        to_add = []
        for special_token in special_tokens:
            if special_token not in words:
                to_add.append(special_token)
        words = to_add + words 
    # store list of words to file
    with open(os.path.join(folder, 'vocab.txt'), 'w') as f:
        for word in words:
            f.write(f'{word}\n')

# make_vocab(
#     train_dataset['sentence'],
#     'our_vocab/',
#     special_tokens=['[PAD]', '[UNK]'],
#     min_freq=2)

# ## HuggingFace's Tokenizer
# tokenizer = BertTokenizer('our_vocab/vocab.txt') # in general we don't need the vocab.txt file
# new_sentence = 'follow the white rabbit neo'
# new_tokens = tokenizer.tokenize(new_sentence) # ['follow', 'the', 'white', 'rabbit', '[UNK]']
# print("Tokens from HF: ", new_tokens) 
# new_ids = tokenizer.convert_tokens_to_ids(new_tokens)
# print("Ids of HF tokens: ", new_ids)
# print("Ids of HF tokens: ", tokenizer.encode(new_sentence, add_special_tokens=False))
# new_ids = tokenizer.encode(new_sentence)
# print("Ids of HF tokens + added special tokens: ", new_ids)
# print("HF tokens + added special tokens: ", tokenizer.convert_ids_to_tokens(new_ids))
# # Checking addtional information delivered by HF tokenizer
# # {'input_ids': tensor([[918,   2, 206, 189,   1]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1]])}
# print("Info from tokenizer of 1 sentence:\n", tokenizer(new_sentence, add_special_tokens=False, return_tensors='pt'))
# sentence1 = 'follow the white rabbit neo'
# sentence2 = 'no one can be told what the matrix is'
# joined_sentences = tokenizer(sentence1, sentence2)
# # {'input_ids': [2194, 918, 1, 206, 189, 1, 2193, 49, 45, 66, 30, 279, 41, 2, 1, 26, 2193], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
# print("Info from tokenizer of 2 sentences:\n", joined_sentences)
# #  ['[CLS]', 'follow', '[UNK]', 'white', 'rabbit', '[UNK]', '[SEP]', 'no', 'one', 'can', 'be', 'told', 'what', 'the', '[UNK]', 'is', '[SEP]']
# print("Convert IDs back to readable tokens:\n", tokenizer.convert_ids_to_tokens(joined_sentences['input_ids']))
# # Tokenize two sentences separately & attention-mask will be 0 at padding positions
# separate_sentences = tokenizer([sentence1, sentence2], padding=True)
# # {'input_ids': [[2194, 918,...], [2194, 49,...]], 'token_type_ids': [[0, 0,..], [0, 0,..]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}
# print("Tokenize 2 sentences:\n", separate_sentences)
# print("Convert IDs of 1st sentence to readable words: ", 
#     tokenizer.convert_ids_to_tokens(separate_sentences['input_ids'][0]))
# print("Attention mask: ", separate_sentences['attention_mask'][0])

# # To create a batch of two sentences each 
# first_sentences = [sentence1, 'another first sentence']
# second_sentences = [sentence2, 'a second sentence here']
# batch_of_pairs = tokenizer(first_sentences, second_sentences)
# first_input = tokenizer.convert_ids_to_tokens(batch_of_pairs['input_ids'][0])
# second_input = tokenizer.convert_ids_to_tokens(batch_of_pairs['input_ids'][1]) 
# print("First batch: ", first_input) # ['[CLS]','follow',...,'[SEP]']
# print("Second batch: ", second_input) # ['[CLS]','another',...,'[SEP]']

# ## Tokenized dataset, padded
# tokenized_dataset = tokenizer(
#     dataset['sentence'],
#     padding = True,
#     return_tensors='pt',
#     max_length=50,
#     truncation=True 
# )
# print("List of sentences' ids:\n", tokenized_dataset['input_ids'])

# ## Bag-of-Words (BoW) - frequency of words
# sentence = 'the white rabbit is a rabbit'
# bow_tokens = simple_preprocess(sentence)
# print('Tokens: ', bow_tokens)
# bow = dictionary.doc2bow(bow_tokens)
# print('BoW of above tokens: ', bow) # [(20,1),(333,2),...]

## Word Embeddings
# Continuous Bag of Words (CBoW): to find vector representation of dictionary
class CBOW(nn.Module):
    # vocab_size: number of rows in dictionary
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        # embedding: huge lookup table of [vocab_size, embedding_size] 
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.linear = nn.Linear(embedding_size, vocab_size)
    # X: context words
    def forward(self, X):
        embeddings = self.embedding(X)
        bow = embeddings.mean(dim=1)
        logits = self.linear(bow)
        return logits 

# torch.manual_seed(42)
# dummy_cbow = CBOW(vocab_size=5, embedding_size=3)
# # Dictionary: ['the','small','is','barking','dog']
# print("Vector representation of dictionary:\n", dummy_cbow.embedding.state_dict())
# print("Vector representation of 'is' (idx=2) and 'barking' (idx=3):\n", dummy_cbow.embedding(torch.as_tensor([2,3])))

# tiny_vocab = ['the', 'small', 'is', 'barking', 'dog']
# context_words = ['the', 'small', 'is', 'barking']
# target_words = ['dog']
# batch_context = torch.as_tensor([[0,1,2,3]]).long() # [1,4]
# batch_target = torch.as_tensor([4]).long() # [1]
# cbow_features = dummy_cbow.embedding(batch_context).mean(dim=1) # [4,3]->mean[1,3]
# print("Context features:", cbow_features)
# logits = dummy_cbow.linear(cbow_features)
# print("Output logits of 5 dict words: ", logits)

# ## Cosine similarity of word embeddings
# ratings = torch.as_tensor([
#     [0.7, -0.4, 0.7],
#     [0.3, 0.7, -0.5],
#     [0.9, -0.55, 0.8],
#     [-0.3, 0.8, 0.34]
# ])
# sims = torch.zeros(4,4)
# for i in range(4):
#     for j in range(4):
#         sims[i,j] = F.cosine_similarity(ratings[i], ratings[j], dim=0)
# print("Similarity confusion matrix:\n", sims)

## Glove - 40000 words, dimension of 50
glove = downloader.load('glove-wiki-gigaword-50')
print("Length of vocabulary: ", len(glove.key_to_index))
print("Embedding of 'alice': ", glove['alice'])
synthetic_queen = glove['king'] - glove['man'] + glove['woman']
fig = plot_word_vectors(
    glove,
    ['king', 'man', 'woman', 'synthetic', 'queen'],
    other={'synthetic': synthetic_queen}
)
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
nltk.download('punkt_tab')


import gensim 
from gensim import corpora, downloader
from gensim.parsing.preprocessing import *
from gensim.utils import simple_preprocess
from datasets import load_dataset, Split
from textattack.augmentation import EmbeddingAugmenter
from transformers import BertTokenizer
from transformers import AutoModel, BertModel, AutoTokenizer, DataCollatorForLanguageModeling
from flair.data import Sentence
from flair.embeddings import ELMoEmbeddings 
from flair.embeddings import WordEmbeddings # like Glove
from flair.embeddings import TransformerWordEmbeddings, TransformerDocumentEmbeddings

from plots.chapter11 import *

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

new_fnames = sentence_tokenize(localfolder)
print("New tokenized .csv files: ", new_fnames)

## HuggingFace
# Loading dataset
dataset = load_dataset(
    path='csv',
    data_files=new_fnames,
    quotechar='\\',
    split=Split.TRAIN 
)
# print("Dataset columns: ", dataset.features) # {'sentence':'','source':''}
# print("Dataset number of columns: ", dataset.num_columns)
# print("Dataset shape: ", dataset.shape)
# print("Dataset sentence number 2: ", dataset[2]) # {'sentence':'xxx', 'source':'alice28.txt'}
# print("Dataset first 3 file-sources: ", dataset['source'][:3])
# print("Dataset source files: ", dataset.unique('source')) # ['a.txt','b.txt']
# Add new column 'labels' to the dataset, 0/1 means belongs to 'alice' source or not
# row: {'sentence': 'xxx', 'source':'alice.txt'}-> return {'label':0 or 1}

def is_alice_label(row):
    is_alice = int(row['source'] == 'alice28-1476.txt')
    return {'labels': is_alice}

dataset = dataset.map(is_alice_label) # {'labels':1,'sentence':'xxx','source':'alice.txt'}
print ("Dataset (added 'labels' column):\n", dataset)
# dataset shuffle and train test split
shuffled_dataset = dataset.shuffle(seed=42)
split_dataset = shuffled_dataset.train_test_split(test_size=0.2)
# {'train':Dataset(), 'test':Dataset()}; each is {'features':['sentence','source','labels'], 'num_rows': 3081}
print("Split dataset:\n", split_dataset) 
train_dataset = split_dataset['train']
test_dataset = split_dataset['test']

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

# ## Glove - 40000 words, dimension of 50
# glove = downloader.load('glove-wiki-gigaword-50')
# print("Length of vocabulary: ", len(glove.key_to_index))
# print("Embedding of 'alice': ", glove['alice'])
# synthetic_queen = glove['king'] - glove['man'] + glove['woman']
# fig = plot_word_vectors(
#     glove,
#     ['king', 'man', 'woman', 'synthetic', 'queen'],
#     other={'synthetic': synthetic_queen}
# )
# plt.savefig('test.png')
# print("Similarity between 'synthetic_queen' and top 5 words:\n", glove.similar_by_vector(synthetic_queen, topn=5))

# # Compare our vocabulary to glove pre-trained word-embeddings
# vocab = list(dictionary.token2id.keys())
# print("Len of our dictionary: ", len(vocab)) # 3706
# # how many and which words are unknown to Glove
# # glove.key_to_index: {'apple':0,'orange':1,...}
# # <set>.difference(<set>) return the keys which are different between 2 sets
# unknown_words = sorted(
#     list(set(vocab).difference(set(glove.key_to_index)))
# )
# print(f"{len(unknown_words)} unknown words, 5 of which are {unknown_words[:5]}")
# unknown_ids = [dictionary.token2id[w]
#     for w in unknown_words
#     if w not in ['[PAD]', '[UNK]']]
# unknown_count = np.sum([dictionary.cfs[idx] for idx in unknown_ids]) # the unknown words occur how many times in the scopus
# print(f"Unknown-embedded words occur {unknown_count} times in our scopus of {dictionary.num_pos} words")

def vocab_coverage(gensim_dict, pretrained_wv, special_tokens=('[PAD]', '[UNK]')):
    vocab = list(gensim_dict.token2id.keys())
    unknown_words = sorted(
        list(set(vocab).difference(set(pretrained_wv.key_to_index)))
    )
    unknown_ids = [gensim_dict.token2id[w]
        for w in unknown_words
        if w not in special_tokens]
    unknown_count = np.sum([gensim_dict.cfs[idx] for idx in unknown_ids])
    cov = 1 - unknown_count / gensim_dict.num_pos 
    return cov 
# print("Coverage of pretrained Glove to our dict: ", vocab_coverage(dictionary, glove))

def make_vocab_from_wv(wv, folder=None, special_tokens=None):
    if folder is not None:
        if not os.path.exists(folder):
            os.mkdir(folder)
    words = wv.index_to_key # ['whiplike', 'breakfront', 'azərbaycan']
    if special_tokens is not None:
        to_add = []
        for special_token in special_tokens:
            if special_token not in words:
                to_add.append(special_token)
        words = to_add + words 
    
    with open(os.path.join(folder, 'vocab.txt'), 'w', encoding='utf-8') as f:
        for word in words:
            f.write(f'{word}\n')

# make_vocab_from_wv(glove, 'glove_vocab/', special_tokens=['[PAD]', '[UNK]'])
# glove_tokenizer = BertTokenizer('glove_vocab/vocab.txt')
# print("Glove encode for world-vector-vocab: ", glove_tokenizer.encode('alice followed the white rabbit', add_special_tokens=False))
# print("Original glove #words: ", len(glove.vectors))
# print("Our newly created #words: ", len(glove_tokenizer.vocab))
# # Add [PAD] and [UNK] to the embeddings list
# special_embeddings = np.zeros((2, glove.vector_size))
# extended_embeddings = np.concatenate(
#     [special_embeddings, glove.vectors], axis=0
# )
# print("New embeddings shape: ", extended_embeddings.shape)
# # 'alice' index in our new tokenizer 
# alice_idx = glove_tokenizer.encode(
#     'alice', add_special_tokens=False
# )
# # check if the glove's embedding and our embeddings match
# print(np.all(extended_embeddings[alice_idx] == glove['alice']))

# ## Model I - Glove + Classifier
# # Data preparation
# train_sentences = train_dataset['sentence']
# train_labels = train_dataset['labels']
# test_sentences = test_dataset['sentence']
# test_labels = test_dataset['labels']
# train_ids = glove_tokenizer(
#     train_sentences,
#     truncation = True, 
#     padding = True,
#     max_length=60,
#     add_special_tokens=False,
#     return_tensors='pt')['input_ids'] # [n_sentences,60]
# train_labels = torch.as_tensor(train_labels).float().view(-1,1)
# test_ids = glove_tokenizer(
#     test_sentences, 
#     truncation=True, 
#     padding=True, 
#     max_length=60, 
#     add_special_tokens=False, 
#     return_tensors='pt')['input_ids']
# test_labels = torch.as_tensor(test_labels).float().view(-1,1)
# train_tensor_dataset = TensorDataset(train_ids, train_labels)
# generator = torch.Generator()
# train_loader = DataLoader(train_tensor_dataset, batch_size=32, shuffle=True, generator=generator)
# test_tensor_dataset = TensorDataset(test_ids, test_labels)
# test_loader = DataLoader(test_tensor_dataset, batch_size=32)
# extended_embeddings = torch.as_tensor(extended_embeddings).float()
# # Pre-trained PyTorch embeddings
# torch_embeddings = nn.Embedding.from_pretrained(extended_embeddings)

# # Check one batch
# token_ids, labels = next(iter(train_loader)) # token_ids: [batch,sentence_len]=[32,60]
# token_embeddings = torch_embeddings(token_ids) # [batch,sentence_len,vector_dim]=[32,60,50]
# print("Average embedding:\n", token_embeddings.mean(dim=1)) # [batch,vector_dim]=[32,50]
# # Bag-of-embeddings (same result, faster)
# boe_mean = nn.EmbeddingBag.from_pretrained(
#     extended_embeddings, mode='mean'
# ) 
# # EmbeddingBag(dict_len,vector_dim,mode='mean')->[400002,50]
# print("Average embedding:\n", boe_mean(token_ids)) # [32,50]
# # Model config and training
# torch.manual_seed(41)
# model = nn.Sequential(
#     boe_mean, # embeddings
#     nn.Linear(boe_mean.embedding_dim, 128),
#     nn.ReLU(),
#     nn.Linear(128,1)
# )
# loss_fn = nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)
# sbs_emb = StepByStep(model, loss_fn, optimizer)
# sbs_emb.set_loaders(train_loader, test_loader)
# sbs_emb.train(20)
# fig = sbs_emb.plot_losses()
# plt.savefig('test.png')
# print("Recall on test set:\n", StepByStep.loader_apply(test_loader, sbs_emb.correct))

## Model II - Glove + Transformer
class TransfClassifier(nn.Module):
    def __init__(self, embedding_layer, encoder, n_outputs):
        super().__init__()
        self.d_model = encoder.d_model 
        self.n_outputs = n_outputs 
        self.embed = embedding_layer 
        self.encoder = encoder 
        self.mlp = nn.Linear(self.d_model, n_outputs)
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, self.d_model)
        )
    
    def preprocess(self, X):
        # [N,L]->[N,L,D]
        src = self.embed(X)
        # [1,1,D]->[N,1,D]
        cls_tokens = self.cls_token.expand(X.size(0), -1, -1)
        # [N,L+1,D]
        src = torch.cat((cls_tokens, src), dim=1)
        return src 
    
    def encode(self, source, source_mask=None):
        states = self.encoder(source, source_mask)
        # 1st hidden state is from CLS token
        cls_state = states[:,0] # [N,D]
        return cls_state 

    @staticmethod 
    def source_mask(X):
        cls_mask = torch.ones(X.size(0), 1).type_as(X) # [N,1]
        # X=0 at padding character
        pad_mask = torch.cat((cls_mask, X>0), dim=1).bool() # [N,1]cat[N,L]->[N,L+1]
        return pad_mask.unsqueeze(1) #[N,1,L+1]

    def forward(self, X):
        # X:[N,L], src:[N,L+1,D]
        src = self.preprocess(X)
        cls_state = self.encode(src, self.source_mask(X))
        out = self.mlp(cls_state)
        return out 

# torch.manual_seed(33)
# # We use bag-of-embeddings only when input is [N,D], but now we need [N,L,D]
# layer = EncoderLayer(
#     n_heads=2, d_model=torch_embeddings.embedding_dim, ff_units=128
# )
# encoder = EncoderTransf(layer, n_layers=1)
# model = TransfClassifier(torch_embeddings, encoder, n_outputs=1)
# loss_fn = nn.BCEWithLogitsLoss()
# optimizer = nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(model.parameters(), lr=1e-4)
# sbs_transf = StepByStep(model, loss_fn, optimizer)
# sbs_transf.set_loaders(train_loader, test_loader)
# sbs_transf.train(10)
# fig = sbs_transf.plot_losses()
# plt.savefig('test.png')
# print("Recall on the test set of TransfClassifier:\n", StepByStep.loader_apply(test_loader, sbs_transf.correct))

# ## Visualizing attention
# sentences = [
#     'The white rabbit and Alice ran away',
#     'The lion met Dorothy on the road'
# ]
# inputs = glove_tokenizer(sentences, add_special_tokens=False, return_tensors='pt')[
#     'input_ids'
# ]
# print("New tokens'ids:\n", inputs)
# # inputs = inputs.to('cuda')
# sbs_transf.model.eval()
# out = sbs_transf.model(inputs)
# print("Those sentences belong to class:\n", torch.sigmoid(out))
# alphas = sbs_transf.model.encoder.layers[0].self_attn_heads.alphas # [N,n_heads,L+1,L+1]->[2,2,8,8]
# print("Attention of the 1st output to the whole sentence(s):\n", alphas[:,:,0,:].squeeze()) # [N,n_heads,1,8]->[N,n_heads,8]
# tokens = [['[CLS]'] + glove_tokenizer.tokenize(sent) for sent in sentences] # [['[CLS]','The','white',...], ['[CLS]','The','lion'...]]
# fig = plot_attention(tokens, alphas)
# plt.savefig('test.png')

## Contextual Word Embeddings 
# ELMo
watch1 = """
The Hatter was the first to break the silence. `What day of the month is it?' he said, turning to Alice:  he had taken his watch out of his pocket, and was looking at it uneasily, shaking it every now and then, and holding it to his ear.
"""
watch2 = """
Alice thought this a very curious thing, and she went nearer to watch them, and just as she came up to them she heard one of them say, `Look out now, Five!  Don't go splashing paint over me like that!
"""
# sentences = [watch1, watch2]
# flair_sentences = [Sentence(s) for s in sentences]
# print("Flair Sentence 0: ", flair_sentences[0])
# # get_token() starts indexing with 1; tokens[] starts indexing at 0 like usual
# print("32nd token: ", flair_sentences[0].get_token(32))
# print("32nd token: ", flair_sentences[0].tokens[31])

# elmo = ELMoEmbeddings()
# print("ELMoEmbeddings of 2 sentences:\n", elmo.embed(flair_sentences)) # [Sentence0, Sentence1]
# token_watch1 = flair_sentences[0].tokens[31] # Token:32 watch
# token_watch2 = flair_sentences[1].tokens[13]  # Token:14 watch
# # each 3072-D tensor([-0.5,-0.42,...]), different, even though same word 'watch'
# print(token_watch1.embedding, token_watch2.embedding) 
# similarity =nn.CosineSimilarity(dim=0, eps=1e-6)
# # similarity 0.6 (tensor)
# print(
#     "Similarity between two words 'watch' from two sentences:\n",
#     similarity(token_watch1.embedding, token_watch2.embedding))

# embeddings: our ELMo or BERT class object
def get_embeddings(embeddings, sentence):
    sent = Sentence(sentence)
    embeddings.embed(sent)
    # stacking [tensor, tensor,...] (58 tensors as 58 words of 3072-D dimension each)
    return torch.stack([
        token.embedding for token in sent.tokens
    ]).float()

# print(get_embeddings(elmo, watch1)) # tensor [58,3072]

# # Explains the 3072 dimensions: adding 6 chunks of 512D each
# # embedding forward + embedding backward + hidden forward layer0 + hidden backward layer0 + hidden forward layer1 + hidden backward layer1
# print("First 2 chunks the same: ", token_watch1.embedding[0], token_watch1.embedding[512])
# print("Two words 'watch' the same: ", (token_watch1.embedding[:1024] == token_watch2.embedding[:1024]).all())

# ## Glove from flair 
# glove_embedding = WordEmbeddings('glove')
# new_flair_sentences = [Sentence(s) for s in sentences] # [Sentence0, Sentence1]
# print("Flair Glove sentences:\n", glove_embedding.embed(new_flair_sentences))
# print("All words 'watch' will have same value:\n", new_flair_sentences[0].tokens[31].embedding == new_flair_sentences[1].tokens[13].embedding) # True

## BERT - Bidirectional Encoder Representations from Transformers
bert_flair = TransformerWordEmbeddings('bert-base-uncased', layers='-1') # using BERT last layer to generate embeddings
embed1 = get_embeddings(bert_flair, watch1)
embed2 = get_embeddings(bert_flair, watch2)
print("Embedding of sentence 1 shape: ", embed1.shape) # [58,768]
print("Embedding of sentence 2 shape: ", embed2.shape) # [48,768]
# Compare the embeddings for the word 'watch' in both sentences
bert_watch1 = embed1[31]
bert_watch2 = embed2[13]
similarity = nn.CosineSimilarity(dim=0, eps=1e-6)
print("Prove that two 'watch' embeddings different:\n", similarity(bert_watch1, bert_watch2))

## Document Embeddings 
documents = [Sentence(watch1), Sentence(watch2)]
bert_doc = TransformerDocumentEmbeddings('bert-base-uncased')
bert_doc.embed(documents) # documents: [Sentence-with-58-tokens, Sentence-with-48-tokens]
# Notice: we don't have single token's embedding anymore, but only 1 document embedding (sentence.tokens[idx] is [])
print("Overall embedding of 1st sentence:\n", documents[0].embedding) # [768]

# embeddings: embedding method Glove or BERT
def get_embeddings(embeddings, sentence):
    sent = Sentence(sentence)
    embeddings.embed(sent)
    # Document embedding
    if len(sent.embedding):
        return sent.embedding.float()
    else:
    # Word embedding: stacking [768D-word-embedding, 768D-word-embedding,...]
        return torch.stack([
            token.embedding for token in sent.tokens 
        ]).float()
print("Get embeddings of 1st sentence:\n", get_embeddings(bert_doc, watch1).shape) # [768]

## Model III - Preprocessed Embeddings 
# Add one more column of features 'embeddings' -> return a Dataset with 4 columns
train_dataset_doc = train_dataset.map(
    lambda row: {'embeddings': get_embeddings(bert_doc, row['sentence'])}
)
test_dataset_doc = test_dataset.map(
    lambda row: {'embeddings': get_embeddings(bert_doc, row['sentence'])}
)
# Set 'embeddings' and 'labels' columns as torch tensors
train_dataset_doc.set_format(type='torch', columns=['embeddings', 'labels'])
test_dataset_doc.set_format(type='torch', columns=['embeddings', 'labels'])

train_dataset_doc = TensorDataset(
    train_dataset_doc['embeddings'].float(),
    train_dataset_doc['labels'].view(-1,1).float()
)
generator = torch.Generator()
train_loader = DataLoader(
    train_dataset_doc, batch_size=32, shuffle=True, generator=generator
)
test_dataset_doc = TensorDataset(
    test_dataset_doc['embeddings'].float(),
    test_dataset_doc['labels'].view(-1,1).float()
)
test_loader = DataLoader(
    test_dataset_doc, batch_size=32, shuffle=True
)

torch.manual_seed(41)
model = nn.Sequential(
    nn.Linear(bert_doc.embedding_length, 3),
    nn.ReLU(),
    nn.Linear(3,1)
)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
sbs_doc_emb = StepByStep(model, loss_fn, optimizer)
sbs_doc_emb.set_loaders(train_loader, test_loader)
sbs_doc_emb.train(20)
fig = sbs_doc_emb.plot_losses()
plt.savefig('test.png')
print("Recall on test dataset:\n", StepByStep.loader_apply(test_loader, sbs_doc_emb.correct))

## BERT
# # AutoModel
# auto_model = AutoModel.from_pretrained('bert-base-uncased') # transformers.modeling_bert.BertModel
bert_model = BertModel.from_pretrained('bert-base-uncased')
print("BERT configs:\n", bert_model.config)
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print("Number of words in BERT Tokenizer:\n", len(bert_tokenizer.vocab)) # 30522
sentence1 = 'Alice is inexplicably following the white rabbit'
sentence2 = 'Follow the white rabbit, Neo'
tokens = bert_tokenizer(sentence1, sentence2, return_tensors='pt')
# {'input_ids': [[101,565,...]], 'token_type_ids':[[0,0,...,1]], 'attention_mask':[[1,1,...,1]]}
print("Bert tokens from two sentences:\n", tokens) 
# ['[CLS]', 'alice', 'is', 'in', '##ex', '##pl', '##ica', '##bly', 'following', 'the', 'white', 'rabbit', '[SEP]', 'follow', 'the', 'white', 'rabbit', ',', 'neo', '[SEP]']
print("Convert Bert tokens to words:\n", bert_tokenizer.convert_ids_to_tokens(tokens['input_ids'][0]))

# # AutoTokenizer
# auto_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased') # transformers.tokenization_bert.BertTokenizer 

# max_sequence_len: 512, embed_dim: 768
# word_embeddings: [30522,768], position_embeddings: [512,768], token_type_embeddings: [2,768]
input_embeddings = bert_model.embeddings 
print("Bert Embeddings:\n", input_embeddings)
token_embeddings = input_embeddings.word_embeddings 
input_token_emb = token_embeddings(tokens['input_ids'])
print("Input embeddings shape: ", input_token_emb.shape) # [1,20,768]
position_embeddings = input_embeddings.position_embeddings 
print("Position embeddings: ", position_embeddings) # Embedding(512,768)
position_ids = torch.arange(512).expand((1,-1)) # [1,512]
seq_length = tokens['input_ids'].size(1) # 20
input_pos_emb = position_embeddings(position_ids[:, :seq_length]) # Embedding(<tensor of [1,20]>)->[1,20,768]
segment_embeddings = input_embeddings.token_type_embeddings # Embedding(2,768) as we have 2 sentences
# Embedding(<tensor of [1,20]>)->[1,20,768]; 
# 1st part is of 1st sentence will have the same embeddings; 2nd part is of 2nd sentence will have same embeddings too
input_seg_emb = segment_embeddings(tokens['token_type_ids']) # [1,20,768]
input_emb = input_token_emb + input_pos_emb + input_seg_emb # [1,20,768]
print(input_emb.shape)

## Masked Language Model (MLM)
# Data Collator
sentence = 'Alice is inexplicably following the white rabbit'
tokens = bert_tokenizer(sentence)
print("MLM tokens' ids:\n", tokens['input_ids'])
torch.manual_seed(41)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=bert_tokenizer, mlm_probability=0.15
)
mlm_tokens = data_collator([tokens])
# {'input_ids': [[101,5650,...,103,...]], 'labels': [[-100,-100,...,5555,...]]}
print("MLM tokens:\n", mlm_tokens) # id=103 & labels=5555 are for the mask '[MASK] replaced token'
print("Original sentence:\n", bert_tokenizer.convert_ids_to_tokens(tokens['input_ids']))
print("Masked sentence:\n", bert_tokenizer.convert_ids_to_tokens(mlm_tokens['input_ids'][0]))

## Next sentence prediction (NSP): predicting the next sentence is the next of input sentence or not 
# Inputs: 2 sentences. Output: 0=fake matched-up; 1=real matched-up)
sentence1 = 'alice follows the white rabbit'
sentence2 = 'follow the white rabbit neo'
print("Bert tokenizing 2 sentences:\n", bert_tokenizer(sentence1, sentence2, return_tensors='pt')) # {'input_ids': [[101,5650,..]], 'token_type_ids':[[0,...,1]], 'attention_mask':[[1,...,1]]}
print("Bert pooler at output:\n", bert_model.pooler) # BertPooler(dense, activation)

## Bert outputs [batch, seq_len, hidden_dimension]
# example: 1st sentence in our Training dataset
sentence = train_dataset[0]['sentence'] # 'And, so far as they knew, they were quite right.'
# {'input_ids':[[101,198,..,0,0]], 'token_type_ids':[[0,..,0]], 'attention_mask':[[1,1,..,0,0]]} 
# => only 'input_ids' and 'attention_mask' are important (we have only 1 sentence=>'token_type_ids' only 1 value)
tokens = bert_tokenizer(
    sentence, padding='max_length', max_length=30, truncation=True, return_tensors='pt'
)
bert_model.eval()
out = bert_model(
    input_ids=tokens['input_ids'],
    attention_mask=tokens['attention_mask'],
    output_attentions=True,
    output_hidden_states=True,
    return_dict=True
)
print("Bert output keys: ", out.keys()) # ['last_hidden_state', 'pooler_output', 'hidden_states', 'attentions']
last_hidden_batch = out['last_hidden_state']
last_hidden_sentence = last_hidden_batch[0] # output full embeddings
# remove [PAD] embeddings
mask = tokens['attention_mask'].squeeze().bool()
embeddings = last_hidden_sentence[mask]
# remove 1st [CLS] & last [SEP]
print("Good embeddings:\n", embeddings[1:-1])
# flair library is doing the same
print("Embedding of that sentence with flair:\n", get_embeddings(bert_flair, sentence))
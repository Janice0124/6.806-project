import sys
import gzip
import random
import numpy as np

# text_tokenized.txt.gz
# maps query IDs to their title and body, body is a list of words
def read_corpus(path):
    raw_corpus = {}
    fopen = gzip.open if path.endswith(".gz") else open
    with fopen(path) as corpus:
        for line in corpus:
            query_id, title, body = line.split("\t")
            title = title.strip().split()
            body = body.strip().split()
            raw_corpus[query_id] = (title, body)
    return raw_corpus

# train_random.txt
# maps query IDs to list of similar IDs and list of negative IDs
def read_train_set(path):
	train_corpus = {}
	with open(path) as txt_file:
		for line in txt_file:
			parts = line.split("\t")
			pid, pos, neg = parts[:3]
			pos = pos.split()
			neg = neg.split()
			train_corpus[pid] = (pos, neg)
	return train_corpus

# Creates train samples using only IDs
def create_id_samples(id_dict):
    '''
	    id_dict: {q: ([p, p], [n, n, n, ...])}
	    for q:
	        for p:
	            create sample: (q, p, 20 random negatives)
    '''
    samples = []
    for qid in id_dict:
        pos, neg = id_dict[qid]
        for p in pos:
            len_neg = len(neg)
            indices = np.random.choice(len_neg, 20, replace=False)
            sample = [qid, p] + [neg[index] for index in indices]
            samples.append(sample)
    return samples

# vectors_pruned.200.txt.gz
# Maps a word to 200-dimension (1D array) feature vector
def read_word_embeddings(path):
	word_embs = {}
	fopen = gzip.open if path.endswith(".gz") else open
	with fopen(path) as corpus:
		for line in corpus:
			parts = line.strip().split(" ")
			word = parts[0]
			vec = np.array([float(v) for v in parts[1:]])
			word_embs[word] = vec
	return word_embs

# Turns a sentence into a feature vector by taking the average
# of all the word embeddings -> outputs 200-dimension 1D array
def line2vec(sentence, word_embeddings):
	feature = np.array([0.0 for i in range(200)])
	num_words = 0
	for word in sentence:
		if word in word_embeddings:
			num_words += 1
			feature += word_embeddings[word]
	return feature / float(num_words) if num_words != 0 else feature

# Creates train samples using actual feature vectors 
# (converts from ID to question to vector)
def create_training_samples(id_samples, word_embs, raw_corpus):
    '''
    for each sample, output tuple of 2 lists: title, body
    [title vector of q: title vector of p, title vector of n's]
    '''
    all_samples = []
    for sample in id_samples:
    	title_sample = []
    	body_sample = []
    	for qid in sample:
    		title, body = raw_corpus[qid]
    		title_sample.append(line2vec(title, word_embs))
    		body_sample.append(line2vec(body, word_embs))
    	all_samples.append((title_sample, body_sample))
    return all_samples

def create_train_batches(batch_size):
	samples = train_samples
	title_batches = []
	body_batches = []
	num_batches = int(len(samples) / batch_size)
	for i in range(num_batches):
		title_batch = []
		body_batch = []
		for j in range(batch_size):
			title, body = samples[i * batch_size + j]
			title_batch.extend(title)
			body_batch.extend(body)
		title_batches.append(title_batch)
		body_batches.append(body_batch)
	return (title_batches, body_batches)

train_ids = read_train_set("../data/askubuntu-master/train_random.txt")
id_samples = create_id_samples(train_ids)
word_embeddings = read_word_embeddings("../data/askubuntu-master/vector/vectors_pruned.200.txt")
raw_corpus = read_corpus("../data/askubuntu-master/text_tokenized.txt.gz")
train_samples = create_training_samples(id_samples, word_embeddings, raw_corpus)
title_batches, body_batches = create_train_batches(20)
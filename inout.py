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



# train_random.txt
# maps query IDs to list of similar IDs and list of negative IDs
def read_train_set(path):
	i = 0
	train_corpus = {}
	with open(path) as txt_file:
		for line in txt_file:
			if i >= 500:
				break
			parts = line.split("\t")
			pid, pos, neg = parts[:3]
			pos = pos.split()
			neg = neg.split()
			train_corpus[pid] = (pos, neg)
			i += 1
	return train_corpus

# Creates train samples of only IDs
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

# Creates train samples using actual feature vectors 
# (converts from ID to question to vector)
def create_samples(id_samples, word_embs, raw_corpus):
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

# Turns list of train samples into batches
# concatenates all lists within each batch
# Returns list of batches, where each batch is [title-batch, body-batch]
def create_train_batches(batch_size, train_samples):
	samples = train_samples
	train_batches = []
	num_batches = int(len(samples) / batch_size)
	for i in range(num_batches):
		title_batch = []
		body_batch = []
		for j in range(batch_size):
			title, body = samples[i * batch_size + j]
			title_batch.extend(title)
			body_batch.extend(body)
		train_batches.append([title_batch, body_batch])
	return train_batches


# end to end function - from files to batches
def build_batches(train_file, dev_file, test_file, word_embs_file, query_corpus_file, batch_size):
    word_embeddings = read_word_embeddings(word_embs_file)
    raw_corpus = read_corpus(query_corpus_file)

    train_ids = read_train_set(train_file)
    id_samples = create_id_samples(train_ids)
    train_samples = create_samples(id_samples, word_embeddings, raw_corpus)
    train_batches = create_train_batches(batch_size, train_samples)

    dev_corpus, dev_id_samples, dev_labs = read_dev_test(dev_file)
    dev_data, dev_labels = create_dev_test_data(dev_id_samples, dev_labs, word_embeddings, raw_corpus)
    # dev_samples = create_samples(dev_id_samples, word_embeddings, raw_corpus)

    test_corpus, test_id_samples, test_labels = read_dev_test(test_file)
    test_data, test_labels = create_dev_test_data(test_id_samples, test_labs, word_embeddings, raw_corpus)
    # test_samples = create_samples(test_id_samples, word_embeddings, raw_corpus)

    return train_batches, dev_samples, dev_labels, test_samples, test_labels

# dev.txt, test.txt
# maps query IDs to list of similar IDs, list of 20 candidates
def read_dev_test(path):
	samples = []
	labels = []
	corpus = {}
	i = 0
	with open(path) as txt_file:
		for line in txt_file:
			if i > 500:
				break
			parts = line.split("\t")
			qid, similar, candidates = parts[:3]
			similar = similar.split()
			candidates = candidates.split()
			corpus[qid] = (similar, candidates)

			samples.append([qid] + candidates)

			s = set(similar)
			sample_labels = [1]
			for c in candidates:
				if c in s:
					sample_labels.append(1)
				else:
					sample_labels.append(0)
			labels.append(sample_labels)
			i+=1
	return (corpus, samples, labels)

def create_dev_test_data(samples, labs, word_embs, raw_corpus):
	title_data = []
	body_data = []
	labels = []
	for i in range(len(samples)):
		if i > 500:
			break
		for qid in samples[i]:
			title, body = raw_corpus[qid]
			title_data.append(line2vec(title, word_embs))
			body_data.append(line2vec(title, word_embs))
		labels.extend(labs[i])

	return ([title_data, body_data], labels)

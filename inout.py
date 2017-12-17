import sys
import gzip
import random
import numpy as np
import tqdm
import torch.utils.data as data

PATH="../data/askubuntu-master/text_tokenized.txt.gz"
DATASET_SIZE = 800

# text_tokenized.txt.gz, corpus-lower.tsv.gz
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

# vectors_pruned.200.txt.gz, glove.pruned.txt.gz
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
def line2vec(sentence, word_embeddings, word_embs_dim):
    feature = np.array([0.0 for i in range(word_embs_dim)])
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
            # if i >= 500:
            #     break
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
def create_samples(id_samples, word_embs, raw_corpus, word_embs_dim):
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
            title_sample.append(line2vec(title, word_embs, word_embs_dim))
            body_sample.append(line2vec(body, word_embs, word_embs_dim))
        all_samples.append((title_sample, body_sample))
    return np.array(all_samples)

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

# ============================================================================
# Part 1
# ============================================================================

# end to end function - from files to batches
def build_batches(train_file, dev_file, test_file, word_embs_file, query_corpus_file, batch_size):
    word_embeddings = read_word_embeddings(word_embs_file)
    raw_corpus = read_corpus(query_corpus_file)
    print "Corpus and embeddings read"

    train_ids = read_train_set(train_file)
    id_samples = create_id_samples(train_ids)
    train_samples = create_samples(id_samples[:5000], word_embeddings, raw_corpus, 200)
    train_batches = create_train_batches(batch_size, train_samples)
    print "Train batches created"

    dev_corpus, dev_id_samples, dev_labs = read_dev_test(dev_file)
    dev_data, dev_labels = create_dev_test_data(dev_id_samples, dev_labs, word_embeddings, raw_corpus)
    # dev_samples = create_samples(dev_id_samples, word_embeddings, raw_corpus)
    print "Dev data created"

    test_corpus, test_id_samples, test_labs = read_dev_test(test_file)
    test_data, test_labels = create_dev_test_data(test_id_samples, test_labs, word_embeddings, raw_corpus)
    # test_samples = create_samples(test_id_samples, word_embeddings, raw_corpus)
    print "Test data created"

    return train_batches, dev_data, dev_labels, test_data, test_labels

# dev.txt, test.txt
# maps query IDs to list of similar IDs, list of 20 candidates
def read_dev_test(path):
    samples = []
    labels = []
    corpus = {}
    i = 0
    with open(path) as txt_file:
        for line in txt_file:
            # if i > 500:
            #     break
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
        # if i > 500:
        #     break
        for qid in samples[i]:
            title, body = raw_corpus[qid]
            title_data.append(line2vec(title, word_embs, 200))
            body_data.append(line2vec(title, word_embs, 200))
        labels.extend(labs[i])

    return ([title_data, body_data], labels)

# ============================================================================
# Part 2
# ============================================================================

def read_eval_Android(pos_file, neg_file, word_embs, android_corpus):
    titles = []
    bodies = []
    labels = []
    with open(pos_file) as pos:
        for line in pos:
            qid, rid = line.strip().split()
            q_title, q_body = android_corpus[qid]
            r_title, r_body = android_corpus[rid]
            titles.append(line2vec(q_title, word_embs, 300))
            bodies.append(line2vec(q_body, word_embs, 300))
            titles.append(line2vec(r_title, word_embs, 300))
            bodies.append(line2vec(r_body, word_embs, 300))
            labels.extend([1,1])
    print "Created Android positives"
    with open(neg_file) as neg:
        for line in neg:
            qid, rid = line.strip().split()
            q_title, q_body = android_corpus[qid]
            r_title, r_body = android_corpus[rid]
            titles.append(line2vec(q_title, word_embs, 300))
            bodies.append(line2vec(q_body, word_embs, 300))
            titles.append(line2vec(r_title, word_embs, 300))
            bodies.append(line2vec(r_body, word_embs, 300))
            labels.extend([0,0])
            if len(titles) > 3000:
                break
    print "Created Android negatives"
    return [titles, bodies], labels

def read_eval_Android2(pos_file, neg_file, word_embs, android_corpus):
    # group pos and neg to query id, then choose 20 neg
    id_to_negs = {}
    with open(neg_file) as neg:
        for line in neg:
            qid, rid = line.strip().split()
            if not id_to_negs.haskey(qid):
                id_to_negs[qid] = []
            id_to_negs[qid].append(rid)

    titles = []
    bodies = []
    labels = []
    with open(pos_file) as pos:
        for line in pos:
            qid, rid = line.strip().split()
            q_title, q_body = android_corpus[qid]
            r_title, r_body = android_corpus[rid]

            len_neg = len(id_to_negs[qid])
            if len_neg >= 20:
                indices = np.random.choice(len_neg, 20, replace=False)
                sample = [qid, rid] + [id_to_negs[qid][index] for index in indices]
            else:
                sample = [qid, rid] + id_to_negs[qid]

            for query in sample:
                title, body = android_corpus[query]
                titles.append(line2vec(title, word_embs, 300))
                bodies.append(line2vec(body, word_embs, 300))
                labels.extend([1, 1] + [0] * 20)
    return [titles, bodies], labels

# ============================================================================
# Part 2 - Direct Transfer
# ============================================================================
'''
    # def batch_test(pos_file, neg_file, word_embs, android_corpus):
    #   id_to_pos = {}
    #   with open(pos_file) as pos:
    #       for line in pos:
    #           qid, rid = line.strip().split()
    #           id_to_pos[qid] = [rid]
    #   return id_to_pos
'''
def build_direct_transfer_data(ubuntu_train_file, android_test_pos_file, android_test_neg_file, word_embs_file, ubuntu_corpus_file, android_corpus_file, batch_size):
    word_embs = read_word_embeddings(word_embs_file)
    print "Read word embeddings"

    raw_corpus = read_corpus(ubuntu_corpus_file)
    print "Read ubuntu corpus"

    train_ids = read_train_set(ubuntu_train_file)
    id_samples = create_id_samples(train_ids)
    print "Created Train samples"

    train_samples = create_samples(id_samples[:1000], word_embs, raw_corpus, 300)
    train_batches = create_train_batches(batch_size, train_samples)
    print "Created Train Batches"

    android_corpus = read_corpus(android_corpus_file)
    test_data, test_labels = read_eval_Android(android_test_pos_file, android_test_neg_file, word_embs, android_corpus)

    return train_batches, test_data, test_labels

# ============================================================================
# Part 2 - Adversarial Domain Adaptation
# ============================================================================
def build_domain_adapt_data(ubuntu_train_file, android_dev_pos_file, android_dev_neg_file, android_test_pos_file, android_test_neg_file, word_embs_file, ubuntu_corpus_file, android_corpus_file, encoder_batch_size, classifier_batch_size):
    
    word_embs = read_word_embeddings(word_embs_file)
    print "Read word embeddings"

    ubuntu_corpus = read_corpus(ubuntu_corpus_file)
    print "Read ubuntu corpus"

    train_ids = read_train_set(ubuntu_train_file)
    id_samples = create_id_samples(train_ids)
    print "Created Ubuntu Train samples"

    train_samples = create_samples(id_samples[:5000], word_embs, ubuntu_corpus, 300)
    train_batches = create_train_batches(encoder_batch_size, train_samples)
    print "Created Ubuntu Train Batches"

    android_corpus = read_corpus(android_corpus_file)
    print "Read android corpus"

    dev_data, dev_labels = read_eval_Android(android_dev_pos_file, android_dev_neg_file, word_embs, android_corpus)
    print "Created android dev data"

    test_data, test_labels = read_eval_Android(android_test_pos_file, android_test_neg_file, 
        word_embs, android_corpus)
    print "Created android test data"

    classifier_data = build_classifier_batches(ubuntu_corpus, android_corpus, word_embs, classifier_batch_size)

    return train_batches, dev_data, dev_labels, test_data, test_labels, classifier_data

'''
def build_classifier_batches(ubuntu_corpus_file, android_corpus_file, word_embs_file, batch_size):
    word_embeddings = read_word_embeddings(word_embs_file)
    ubuntu_corpus = read_corpus(ubuntu_corpus_file)
    android_corpus = read_corpus(android_corpus_file)

    ubuntu_keys = ubuntu_corpus.keys()
    android_keys = android_corpus.keys()

    min_size = min(len(ubuntu_corpus), len(android_corpus))
    for i in range(int(min_size / batch_size)):
        for j in range(batch_size):
            index = i * batch_size + j
'''
def build_classifier_batches(ubuntu_corpus, android_corpus, word_embs, classifier_batch_size):
    '''
    builds batches of size 40, with half ubuntu and half android
    '''
    batch_size = classifier_batch_size/2

    ubuntu_keys = ubuntu_corpus.keys()
    android_keys = android_corpus.keys()

    title_batches = []
    body_batches = []
    label_batches = []

    min_size = min(len(ubuntu_corpus), len(android_corpus))
    min_size = 1000 # remove

    for i in range(int(min_size)/batch_size):
        title_batch = []
        body_batch = []
        label_batch = []
        for j in range(batch_size):
            index = i * batch_size + j
            u_qid = ubuntu_keys[index]
            u_title, u_body = ubuntu_corpus[u_qid]
            a_qid = android_keys[index]
            a_title, a_body = android_corpus[a_qid]

            title_batch.append(line2vec(u_title, word_embs, 300))
            body_batch.append(line2vec(u_body, word_embs, 300))
            label_batch.append(0)

            title_batch.append(line2vec(a_title, word_embs, 300))
            body_batch.append(line2vec(a_body, word_embs, 300))
            label_batch.append(1)
        title_batches.append(title_batch)
        body_batches.append(body_batch)
        label_batches.append(label_batch)
    return (title_batches, body_batches, label_batches)

    # ubuntu_questions = []
    # android_questions = []
    # for question in ubuntu_corpus:
    #     ubuntu_questions.append(ubuntu_corpus[question])
    # for question in android_corpus:
    #     android_questions.append(android_corpus[question])

    # # print ubuntu_questions[:10]

    # ubuntu_keys = ubuntu_corpus.keys()
    # android_keys = android_corpus.keys()

    # body_batches = {'bodies': [], 'labels': []}
    # title_batches = {'titles': [], 'labels': []}

    # min_size = min(len(ubuntu_corpus), len(android_corpus)) 
    # min_size = 1000 # remove this

    # for i in range(int(min_size / batch_size)): # num batches = 40
    #     new_body_batch = []
    #     new_title_batch = []
    #     new_label_body_batch = []
    #     new_label_title_batch = []
    #     for j in range(batch_size): # for question in batch
    #         index = i * batch_size + j

    #         new_body_batch.append(line2vec(ubuntu_questions[index][1], word_embs, 300))
    #         new_label_body_batch.append(0)
    #         new_body_batch.append(line2vec(android_questions[index][1], word_embs, 300))
    #         new_label_body_batch.append(1)

    #         new_title_batch.append(line2vec(ubuntu_questions[index][0], word_embs, 300))
    #         new_label_title_batch.append(0)
    #         new_title_batch.append(line2vec(android_questions[index][0], word_embs, 300))
    #         new_label_title_batch.append(1)

    #         if len(new_body_batch) > 40:
    #             body_batches['bodies'].append(new_body_batch)
    #             body_batches['labels'].append(new_label_body_batch)
    #             title_batches['titles'].append(new_title_batch)
    #             title_batches['labels'].append(new_label_title_batch)
    #             break
    # return (body_batches, title_batches)


android_neg_dev = "../data/android-master/dev.neg.txt"
android_pos_dev = "../data/android-master/dev.pos.txt"
android_neg_test = "../data/android-master/test.neg.txt"
android_pos_test = "../data/android-master/test.pos.txt"
android_corpus = "../data/android-master/corpus-lower.tsv"
glove_embeddings = "../data/android-master/glove.pruned.txt"
query_corpus_file = "../data/askubuntu-master/text_tokenized.txt.gz"
# build_classifier_batches(query_corpus_file, android_corpus, glove_embeddings, 40)

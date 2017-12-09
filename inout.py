import sys
import gzip
import random
import numpy as np

# text_tokenized.txt.gz
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

# vectors_pruned.200.txt.gz
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

def sentence2vec(sentence, word_embeddings):
	words = sentence.split(" ")
	feature = [0.0 for i in range(200)]
	num_words = 0
	for word in words:
		if word in word_embeddings:
			num_words += 1
			feature += word_embeddings[word]
	return feature / float(num_words)

def load_embedding_iterator(path):
    file_open = gzip.open if path.endswith(".gz") else open
    with file_open(path) as emb_file:
        for line in emb_file:
            line = line.strip()
            if line:
                parts = line.split()
                word = parts[0]
                values = np.array([ float(x) for x in parts[1:] ])
                yield word, values




class EmbeddingLayer(object):
    '''
        Embedding layer that
                (1) maps string tokens into integer IDs
                (2) maps integer IDs into embedding vectors (as matrix)
        Inputs
        ------
        vocab           : an iterator of string tokens; the layer will allocate an ID
                            and a vector for each token in it
        oov             : out-of-vocabulary token
        embs            : an iterator of (word, vector) pairs; these will be added to
                            the layer
    '''
    def __init__(self, vocab, oov="<unk>", embs=None):
        if embs is not None:
            lst_words = []
            vocab_map = {}
            emb_vals = []
            for word, vector in embs:
                vocab_map[word] = len(vocab_map)
                emb_vals.append(vector)
                lst_words.append(word)

            for word in vocab:
                if word not in vocab_map:
                    vocab_map[word] = len(vocab_map)
                    emb_vals.append(random_init((200,))*(0.001 if word != oov else 0.0))
                    lst_words.append(word)

            emb_vals = np.vstack(emb_vals)
            self.vocab_map = vocab_map
            self.lst_words = lst_words
        else:
            lst_words = [ ]
            vocab_map = {}
            for word in vocab:
                if word not in vocab_map:
                    vocab_map[word] = len(vocab_map)
                    lst_words.append(word)

            self.lst_words = lst_words
            self.vocab_map = vocab_map
            emb_vals = random_init((len(self.vocab_map), 200))
            self.init_end = -1

        if oov is not None and oov is not False:
            assert oov in self.vocab_map, "oov {} not in vocab".format(oov)
            self.oov_tok = oov
            self.oov_id = self.vocab_map[oov]
        else:
            self.oov_tok = None
            self.oov_id = -1

        self.embeddings = create_shared(emb_vals)
        if self.init_end > -1:
            self.embeddings_trainable = self.embeddings[self.init_end:]
        else:
            self.embeddings_trainable = self.embeddings

        self.n_V = len(self.vocab_map)
        self.n_d = 200

    def map_to_words(self, ids):
        n_V, lst_words = self.n_V, self.lst_words
        return [ lst_words[i] if i < n_V else "<err>" for i in ids ]

    def map_to_ids(self, words, filter_oov=False):
        '''
            map the list of string tokens into a numpy array of integer IDs
            Inputs
            ------
            words           : the list of string tokens
            filter_oov      : whether to remove oov tokens in the returned array
            Outputs
            -------
            return the numpy array of word IDs
        '''
        vocab_map = self.vocab_map
        oov_id = self.oov_id
        if filter_oov:
            not_oov = lambda x: x!=oov_id
            return np.array(
                    filter(not_oov, [ vocab_map.get(x, oov_id) for x in words ]),
                    dtype="int32"
                )
        else:
            return np.array(
                    [ vocab_map.get(x, oov_id) for x in words ],
                    dtype="int32"
                )

    def forward(self, x):
        '''
            Fetch and return the word embeddings given word IDs x
            Inputs
            ------
            x           : an array of integer IDs
            Outputs
            -------
            a matrix of word embeddings
        '''
        return self.embeddings[x]

    @property
    def params(self):
        return [ self.embeddings_trainable ]

    @params.setter
    def params(self, param_list):
        self.embeddings.set_value(param_list[0].get_value())

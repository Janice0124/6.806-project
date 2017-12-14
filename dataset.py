import sys
import gzip
import random
import numpy as np
import tqdm
import torch.utils.data as data
import inout


PATH="../data/askubuntu-master/text_tokenized.txt.gz"
DATASET_SIZE = 800

def getEmbeddingTensor():

    embedding_path='../data/askubuntu-master/vector/vectors_pruned.200.txt.gz'
    lines = []
    with gzip.open(embedding_path) as file:
        lines = file.readlines()
        file.close()
    embedding_tensor = []
    word_to_indx = {}
    for indx, l in enumerate(lines):
        word, emb = l.split()[0], l.split()[1:]
        vector = [float(x) for x in emb ]
        if indx == 0:
            embedding_tensor.append( np.zeros( len(vector) ) )
        embedding_tensor.append(vector)
        word_to_indx[word] = indx+1
    embedding_tensor.append(np.zeros(len(vector)))
    embedding_tensor = np.array(embedding_tensor, dtype=np.float32)
    return embedding_tensor, word_to_indx

def load_dataset(args):
    print("\nLoading data...")
    embeddings, word_to_indx = getEmbeddingTensor()
    args.embedding_dim = embeddings.shape[1]

    train_data = dataset.Dataset('train', word_to_indx)
    dev_data = dataset.Dataset('heldout', word_to_indx)

    return train_data, dev_data, embeddings


class Dataset(data.Dataset):

    def __init__(self, name, word_to_indx, max_length=50, stem="../data/askubuntu-master/text_tokenized"):
        self.name = name
        self.dataset = []
        self.word_to_indx  = word_to_indx
        self.max_length = max_length
        self.raw_corpus = inout.read_corpus(PATH)

        with gzip.open(PATH.format(name)) as gfile:
            lines = gfile.readlines()[:DATASET_SIZE]
            for line in tqdm.tqdm(lines):
                sample = self.processLine(line)
                self.dataset.append(sample)
            gfile.close()


    ## EDIT THIS PART
    def processLine(self, line):
        labels = [ float(v) for v in line.split()[:5] ]

        label = float(labels[0])
        text = line.split('\t')[-1].split()[:self.max_length]
        x =  getIndicesTensor(text, self.word_to_indx, self.max_length)
        sample = {'x':x, 'y':label}
        return sample

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,index):
        sample = self.dataset[index]
        return sample


def getIndicesTensor(text_arr, word_to_indx, max_length):
    unk_indx = 0
    text_indx = [ word_to_indx[x] if x in word_to_indx else unk_indx for x in text_arr][:max_length]
    if len(text_indx) < max_length:
        text_indx.extend( [nil_indx for _ in range(max_length - len(text_indx))])

    x =  torch.LongTensor(text_indx)

    return x

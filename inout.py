import sys
import gzip
import random

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

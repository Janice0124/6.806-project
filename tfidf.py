from sklearn.feature_extraction.text import TfidfVectorizer
import inout as utils
import gzip
import numpy as np
import meter
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

android_corpus_file = "../data/Android-master/corpus-lower.tsv.gz"
android_test_pos = "../data/Android-master/test.pos.txt"
android_test_neg = "../data/Android-master/test.neg.txt"

path = android_corpus_file
fopen = gzip.open if path.endswith(".gz") else open
lines = []
id_to_index = {}
i = 0
with fopen(path) as corpus:
	print "Reading query corpus"
	for line in corpus:
	    query_id, title, body = line.split("\t")
	    lines.append(title.strip() + " " + body.strip())
	    id_to_index[query_id] = i
	    i += 1

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(lines).toarray()

print "Creating meter"
m = meter.AUCMeter()

cos_sims = []
labels = []
with open(android_test_pos) as test_file:
	print "Reading test.pos"
	for line in test_file:
		qid, rid = line.strip().split(" ")
		q_vec = vectors[id_to_index[qid]]
		r_vec = vectors[id_to_index[rid]]
		q_emb = Variable(torch.FloatTensor(q_vec))
		r_emb = Variable(torch.FloatTensor(r_vec))
		cos_sim = F.cosine_similarity(q_emb, r_emb, dim=0, eps=1e-6)
		cos_sims.append(cos_sim.data[0])
		labels.append(1)
print "Adding positive output and target to meter"
print cos_sims
m.add(torch.FloatTensor(cos_sims), torch.IntTensor(labels))

with open(android_test_neg) as test_file:
	cos_sims = []
	labels = []
	i = 0
	print "Reading test.neg"
	for line in test_file:
		qid, rid = line.strip().split(" ")
		q_vec = vectors[id_to_index[qid]]
		r_vec = vectors[id_to_index[rid]]
		q_emb = Variable(torch.FloatTensor(q_vec))
		r_emb = Variable(torch.FloatTensor(r_vec))
		cos_sim = F.cosine_similarity(q_emb, r_emb, dim=0, eps=1e-6)
		cos_sims.append(cos_sim.data[0])
		labels.append(0)
		i += 1
		if i % 1000 == 0:
			print "index: ", i
			print cos_sims
			m.add(torch.FloatTensor(cos_sims), torch.IntTensor(labels))
			cos_sims = []
			labels = []
print m.value(max_fpr=0.05)

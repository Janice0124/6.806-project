from sklearn.feature_extraction.text import TfidfVectorizer
import inout as utils
import gzip
import numpy as np
import meter
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

from sklearn import metrics
import torch.optim as optim
import torch.utils.data
from meter import AUCMeter
from evaluation import Evaluation
from tqdm import tqdm


ubuntu_train_file = "../data/askubuntu-master/train_random.txt"
ubuntu_corpus_file = "../data/askubuntu-master/text_tokenized.txt.gz"

android_corpus_file = "../data/android-master/corpus-lower.tsv.gz"
glove_embeddings = "../data/android-master/glove.pruned.txt.gz"
android_test_pos = "../data/Android-master/test.pos.txt"
android_test_neg = "../data/Android-master/test.neg.txt"

train_data, test_data, test_labels = utils.build_direct_transfer_data(ubuntu_train_file, android_test_pos, android_test_neg, glove_embeddings, ubuntu_corpus_file, android_corpus_file, 20)
print "Created train and test data"

class DAN(nn.Module):

    def __init__(self, embeddings, args):
        super(DAN, self).__init__()
        self.input_dim = args[0]
        # self.embedding_layer = nn.Embedding(len(embeddings), len(embeddings[0]))
        self.seq = nn.Sequential(
                nn.Linear(self.input_dim, 200),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(200,100), # try dropout layer w/ varying probabilities, weight decay
                nn.Tanh())

    def forward(self, x):
        # x = self.embedding_layer(Variable(torch.FloatTensor(x)))
        # x = torch.mean(x, dim=1)
        x = self.seq(x)
        return x

def train(model, train_data, max_epoches, verbose=False):
    model.train()
    weight_decay = 1e-5 # 1e-5
    lr = 1e-3 # 1e-3
    dc_lr = 1e-3
    l = 1e-3 #lambda
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    domain_classifier_optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MultiMarginLoss(margin=0.2)
    criterion_da = nn.BCELoss()
    best_dev = 0.0
    corresponding_test = 0.0
    loss_count = 0
    loss_range = 0.1
    prev_loss = 0

    for epoch in range(max_epoches):
        print "==============="
        print "EPOCH ", epoch
        batch_num = 0
        for batch in train_data:
            batch_num+=1

            titles, bodies = batch
            title_embeddings = Variable(torch.FloatTensor(titles))
            body_embeddings = Variable(torch.FloatTensor(bodies))
            title_output = model(title_embeddings)
            body_output = model(body_embeddings)
            question_embeddings = (title_output + body_output)/2.

            X = []
            for i in range(20):
                query_emb = question_embeddings[i * 20]
                for j in range(22):
                    if j != 0:
                        index = i * 20 + j
                        if query_emb.size()!=(100L,1L):
                            query_emb=torch.unsqueeze(query_emb, 1) 
                        question_embeddings_index = torch.unsqueeze(question_embeddings[index], 1) 
                        X.append(F.cosine_similarity(torch.t(query_emb), torch.t(question_embeddings_index), dim=1))

            Y = np.array([0 for i in range(20)])
            
            optimizer.zero_grad()

            loss = criterion(torch.cat(X), Variable(torch.LongTensor(Y)))
            loss.backward()
            optimizer.step()
        print "Loss", loss
        # if (abs(loss.data[0] - prev_loss <= loss_range)):
        # 	loss_count += 1
        # else:
        # 	loss_count = 0
        # prev_loss = loss.data[0]
        # if loss_count >= 5:
        # 	break
        print "=============="

        # evaluate(dev_data, dev_labels, model) # evaluate on android dataset

def evaluate(model, test_data, test_labels):
	m = AUCMeter()
	cos_sims = []
	labels = []
	titles, bodies = test_data
	print "Getting test query embeddings"
	title_output = model(Variable(torch.FloatTensor(titles)))
	body_output = model(Variable(torch.FloatTensor(bodies)))
	question_embeddings = (title_output + body_output)/2
	print "Getting cosine similarities"
	for i in range(len(question_embeddings)/2):
		q_ind = 2 * i
		r_ind = 2 * i + 1
		q_emb = question_embeddings[q_ind]
		r_emb = question_embeddings[r_ind]
		cos_sim = F.cosine_similarity(q_emb, r_emb, dim=0, eps=1e-6)
		cos_sims.append(cos_sim.data[0])
		labels.append(test_labels[q_ind])
		if i % 3000 == 0 or i == len(question_embeddings)/2:
			print "index ", q_ind
	m.add(torch.FloatTensor(cos_sims), torch.IntTensor(labels))
	print m.value(max_fpr=0.05)



torch.manual_seed(1)
model = DAN(train_data, [300])
train(model, train_data, 20)
evaluate(model, test_data, test_labels)



'''
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

print m.value(max_fpr=0.05)
'''
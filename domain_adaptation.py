import numpy as np
from sklearn import metrics
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import gzip
import inout as utils
from meter import AUCMeter
from sklearn.feature_extraction.text import TfidfVectorizer
from evaluation import Evaluation
from tqdm import tqdm
import meter

word_embs_file = "../data/android-master/glove.pruned.txt"

train_file = "../data/askubuntu-master/train_random.txt"
dev_file = "../data/askubuntu-master/dev.txt"
test_file = "../data/askubuntu-master/test.txt"

ubuntu_corpus_file = "../data/askubuntu-master/text_tokenized.txt.gz"
android_corpus_file = "../data/android-master/corpus-lower.tsv.gz"

android_pos_dev = "../data/android-master/dev.pos.txt"
android_neg_dev = "../data/android-master/dev.neg.txt"
android_pos_test = "../data/android-master/test.pos.txt"
android_neg_test = "../data/android-master/test.neg.txt"

torch.manual_seed(1)
batch_size = 20
hidden_dim = 300

# train_batches, dev_data, dev_labels, test_data, test_labels = utils.build_batches(train_file, dev_file, test_file, word_embs_file, ubuntu_corpus_file, 20)

# classifier_batches = utils.build_classifier_batches(ubuntu_corpus_file, android_corpus_file, word_embs_file, 40)
# (android_titles_bodies, android_labels) = utils.read_eval_Android(android_pos_dev,android_neg_dev,glove_embeddings,android_corpus)
# dev_data_android = android_titles_bodies
# dev_labels_android = android_labels

class DAN(nn.Module):

    def __init__(self, embeddings, args):
        super(DAN, self).__init__()
        self.input_dim = args[0]
        # self.embedding_layer = nn.Embedding(len(embeddings), len(embeddings[0]))
        self.seq = nn.Sequential(
                nn.Linear(self.input_dim, 200),
                nn.ReLU(),
                nn.Dropout(p=0.05),
                nn.Linear(200,100), # try dropout layer w/ varying probabilities, weight decay
                nn.Tanh())

    def forward(self, x):
        # x = self.embedding_layer(Variable(torch.FloatTensor(x)))
        # x = torch.mean(x, dim=1)
        x = self.seq(x)
        return x

class DomainClassifier(nn.Module):
    def __init__(self, embeddings, args):
        super(DomainClassifier, self).__init__()
        self.args = args
        self.seq = nn.Sequential(
                nn.Linear(100, 300),
                nn.ReLU(),
                nn.Linear(300, 150),
                nn.ReLU(),
                nn.Linear(150,1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.seq(x)
        # print x.size()
        x = torch.squeeze(x)
        x = self.sigmoid(x)
        return x

def train(dan_model, train_data, max_epoches, dev_data_android, dev_labels_android, classifier_data, verbose=False):
    dan_model.train()
    weight_decay = 1e-5# 1e-5
    lr = 1e-3# 1e-3
    dc_lr = 1e-3
    l = 1e-5 #lambda
    # weight_decay = 1
    # dropout = 0.3
    # lr = 0.0005
    optimizer = optim.Adam(dan_model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.MultiMarginLoss(margin=0.2)
    
    da_titles, da_bodies, da_labels = classifier_data
    da_labels = np.asarray(da_labels)
    da_train_batches = []
    for i in range(len(da_bodies)):
        da_train_batches.append([da_bodies[i], da_titles[i]])
    da_model = DomainClassifier(da_train_batches, [])
    da_model.train()
    domain_classifier_optimizer = optim.Adam(da_model.parameters(), lr=lr)
    criterion_da = nn.BCELoss()

    best_dev = 0.0
    corresponding_test = 0.0

    for epoch in range(max_epoches):
        print "epoch", epoch
        batch_num = 0
        for batch in train_data:
            # print batch_num
            batch_num+=1
            titles, bodies = batch
            title_embeddings = Variable(torch.FloatTensor(titles))
            body_embeddings = Variable(torch.FloatTensor(bodies))
            title_output = dan_model(title_embeddings)
            body_output = dan_model(body_embeddings)
            question_embeddings = (title_output + body_output)/2.
            # len(question_embeddings) = 440 = 22 * 20
            '''
            create matrix by iterating from 0 to 20, 0 to 21:
            x = 20x21 matrix, mapping q to cosine similarity of each of 21 questions for each set of 22 questions
            y = list of positive question indices, which is always 0 in that row
            '''
            X = []
            for i in range(20):
                query_emb = question_embeddings[i * 20]
                for j in range(22):
                    if j != 0:
                        index = i * 20 + j
                        if query_emb.size()!=(100L,1L):
                            query_emb=torch.unsqueeze(query_emb, 1) 
                        question_embeddings_index = torch.unsqueeze(question_embeddings[index], 1) 
                        # X[i, j-1] = F.cosine_similarity(query_emb, question_embeddings[index], dim=1)
                        X.append(F.cosine_similarity(torch.t(query_emb), torch.t(question_embeddings_index), dim=1))

            Y = np.array([0 for i in range(20)])
            
            optimizer.zero_grad()

            loss = criterion(torch.cat(X), Variable(torch.LongTensor(Y)))
            # print "loss", loss
            
            '''domain adaptation'''
            domain_classifier_optimizer.zero_grad()

            title_embeddings = Variable(torch.FloatTensor(da_titles))
            body_embeddings = Variable(torch.FloatTensor(da_bodies))
            title_output = dan_model(title_embeddings)
            body_output = dan_model(body_embeddings)
            question_embeddings = (title_output + body_output)/2.
            question_labels = da_model(question_embeddings)
            # print type(question_labels), type(labels)
            # print question_labels, labels
            da_loss = criterion_da((question_labels), Variable(torch.FloatTensor(da_labels)))
            # print da_loss
            # print l*da_loss, loss
            total_loss = loss - l*da_loss
            total_loss.backward()
            optimizer.step()
            domain_classifier_optimizer.step()

        # evaluate(dev_data_android, dev_labels_android, dan_model)
        get_auc(dev_data_android, dev_labels_android, dan_model)

        # dev_title_output = model(dev_title_embs)
        # dev_body_output = model(dev_body_embs)
        # dev_question_output = np.mean(dev_title_output, dev_body_output, axis=0)
    

        # dev = evaluate(model, dev_loader)
    #     dev = evaluate(model, dev_data)
    #     # test = evaluate(model, test_loader)
    #     test = evaluate(model, test_data)
    #     if dev > best_dev:
    #         best_dev = dev
    #         corresponding_test = test

    # print (best_dev, corresponding_test)

def get_auc(data, labels, model):
    m = meter.AUCMeter()
    scores = []
    titles, bodies = data
    # score_labels = []
    # for i in range(len(labels)/2):
    #     q = 2 * i
    #     r = 2 * i + 1
    #     title_q = titles[q]
    #     body_q = bodies[q]
    #     title_q_emb = Variable(torch.FloatTensor(title_q))
    #     body_q_emb = Variable(torch.FloatTensor(body_q))
    #     title_q_output = model(title_q_emb)
    #     body_q_output = model(body_q_emb)
    #     q_emb = (title_q_output + body_q_output)/2.

    #     title_r = titles[r]
    #     body_r = bodies[r]
    #     title_r_emb = Variable(torch.FloatTensor(title_r))
    #     body_r_emb = Variable(torch.FloatTensor(body_q))
    #     title_r_out = model(title_r_emb)
    #     body_r_out = model(body_r_emb)
    #     r_emb = (title_r_out + body_r_out)/2.

    #     cos_sim = F.cosine_similarity(q_emb, r_emb, dim=0, eps=1e-6)
    #     scores.append(cos_sim.data[0])
    #     score_labels.append(labels[r])


    # scores = np.asarray(scores)
    # score_labels = np.asarray(score_labels)
    # print scores.shape, score_labels.shape
    # test = torch.FloatTensor(scores)
    # test1 = torch.FloatTensor(score_labels)
    # m.add(scores, score_labels)
    # print m.value(max_fpr=0.05)

    for i in range(len(labels)):
        titles_i = titles[i]
        bodies_i = bodies[i]
        labels_i = labels[i]
        title_embeddings = Variable(torch.FloatTensor(titles_i))
        body_embeddings = Variable(torch.FloatTensor(bodies_i))
        title_output = model(title_embeddings)
        body_output = model(body_embeddings)
        question_embeddings = (title_output + body_output)/2.
        question_embeddings_query = torch.unsqueeze(question_embeddings[0], 1) 
        question_embeddings_candidates = torch.unsqueeze(question_embeddings[1:], 1) 
        # print question_embeddings_query.size(), question_embeddings_candidates.size()
        scores.append(F.cosine_similarity(torch.t(question_embeddings_query), torch.t(question_embeddings_candidates)).data.cpu().numpy()[0])
    # print type(scores), type(labels), type(scores[0]), type(labels[0])

    scores = np.asarray(scores)
    labels = np.asarray(labels)
    print scores.shape, labels.shape
    test = torch.FloatTensor(scores)
    test1 = torch.FloatTensor(labels)
    m.add(scores, labels)
    print m.value(max_fpr=0.05)

def evaluate(data, labels, model):
    res = [ ]
    model.eval()
    res = compute_scores(data, labels, model)
    evaluation = Evaluation(res)
    MAP = evaluation.MAP()*100
    MRR = evaluation.MRR()*100
    P1 = evaluation.Precision(1)*100
    P5 = evaluation.Precision(5)*100
    print MAP, MRR, P1, P5
    return MAP, MRR, P1, P5

def compute_scores(data, labels, model):
    res = []
    scores = []
    curr_labels = []
    titles = data[0]
    bodies = data[1]
    # print len(titles), len(bodies), len(labels)
    # print titles[0], titles[1], " space", bodies[0], bodies[1], 'space', labels[0], labels[1]
    for i in range(len(labels)):
        curr_labels.append(labels[i])
        if i%21==0 and i != 0:
            # print "len curr", len(curr_labels)
            ranks=np.asarray(scores)
            ranks=ranks.argsort()
            # print ranks
            # print np.asarray(curr_labels)
            ranked_labels = np.asarray(curr_labels)[-ranks]
            # print ranked_labels
            res.append(ranked_labels)
            scores = []
            curr_labels = []
        titles_i = titles[i]
        bodies_i = bodies[i]
        labels_i = labels[i]
        # print (titles_i), (bodies_i), (labels_i), 'titles bodies labels'
        
        title_embeddings = Variable(torch.FloatTensor(titles_i))
        body_embeddings = Variable(torch.FloatTensor(bodies_i))
        title_output = model(title_embeddings)
        body_output = model(body_embeddings)
        question_embeddings = (title_output + body_output)/2.
        question_embeddings_query = torch.unsqueeze(question_embeddings[0], 1) 
        question_embeddings_candidates = torch.unsqueeze(question_embeddings[1:], 1) 
        # print question_embeddings_query.size(), question_embeddings_candidates.size()
        scores.append(F.cosine_similarity(torch.t(question_embeddings_query), torch.t(question_embeddings_candidates)).data.cpu().numpy()[0])
        
    return res



# model = DAN(train_batches, [])
# model = LSTM(train_batches, [])


android_corpus = utils.read_corpus(android_corpus_file)
glove_embeddings = utils.read_word_embeddings(word_embs_file)

ubuntu_train_batches, dev_data_android, dev_labels_android, test_data_android, test_labels_android, classifier_data = utils.build_domain_adapt_data(train_file, android_pos_dev, android_neg_dev, android_pos_test, android_neg_test, word_embs_file, ubuntu_corpus_file, android_corpus_file, 20, 40)
dan_model = DAN(ubuntu_train_batches, [300])

train(dan_model, ubuntu_train_batches, 5, dev_data_android, dev_labels_android, classifier_data, verbose=False)
print "now testing"
get_auc(test_data_android, test_labels_android, dan_model)
#CHANGE NUM EPOCHS


'''
A) scikit learn tfidf TfidfVectorizer
put in a sentence into tfidf -> output vector, gives diff type of representation
cosine sims eval

B) 

2) Make domain classifier model, softmax, generates 0 or 1 (ubuntu or android)
Loss = loss1-1e-3 loss2 (want loss2 to be bad). Train encoder on both losses
Feed in batch 1: same as part 1
Batch 2: nrandom ubuntu, nrandom android

Report best hyperparams
'''

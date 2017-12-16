import numpy as np
from sklearn import metrics
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import gzip
import inout
from meter import AUCMeter
from sklearn.feature_extraction.text import TfidfVectorizer
from evaluation import Evaluation
from tqdm import tqdm

train_file = "../data/askubuntu-master/train_random.txt"
dev_file = "../data/askubuntu-master/dev.txt"
test_file = "../data/askubuntu-master/test.txt"
word_embs_file = "../data/askubuntu-master/vector/vectors_pruned.200.txt"
query_corpus_file = "../data/askubuntu-master/text_tokenized.txt.gz"

android_neg_dev = "../data/android-master/dev.neg.txt"
android_pos_dev = "../data/android-master/dev.pos.txt"
android_neg_test = "../data/android-master/test.neg.txt"
android_pos_test = "../data/android-master/test.pos.txt"
android_corpus_file = "../data/android-master/corpus-lower.tsv.gz"
glove_embeddings_file = "../data/android-master/glove.pruned.txt"

torch.manual_seed(1)
batch_size = 20
hidden_dim = 300


# train_x, train_y = read_file("../data/askubuntu-master/train_random.txt")
# dev_x, dev_y = read_file("../data/askubuntu-master/dev.txt")
# test_x, test_y = read_file("../data/askubuntu-master/test.txt")
# f = open('../data/askubuntu-master/vector/vectors_pruned.200.txt', 'r')

# train_x = extract_features(train_x)
# dev_x = extract_features(dev_x)
# test_x = extract_features(test_x)

android_corpus = inout.read_corpus(android_corpus_file)
glove_embeddings = inout.read_word_embeddings(glove_embeddings_file)
train_batches, dev_data, dev_labels, test_data, test_labels = inout.build_batches(train_file, dev_file, test_file, word_embs_file, query_corpus_file, 20)
classifier_batches = inout.build_classifier_batches(query_corpus_file, android_corpus_file, glove_embeddings_file, 40)
(android_titles_bodies, android_labels) = inout.read_eval_Android(android_pos_dev,android_neg_dev,glove_embeddings,android_corpus)
dev_data_android = android_titles_bodies
dev_labels_android = android_labels
# train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(train_x), torch.LongTensor(train_y))
# dev_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(dev_x), torch.LongTensor(dev_y))
# test_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(test_x), torch.LongTensor(test_y))

# train_loader = torch.utils.data.DataLoader(train_dataset)
# dev_loader = torch.utils.data.DataLoader(dev_dataset)
# test_loader = torch.utils.data.DataLoader(test_dataset)

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

class LSTM(nn.Module):
    def __init__(self, embeddings, args):
        super(LSTM, self).__init__()
        self.args = args
        self.lstm = nn.LSTM(input_size=200, hidden_size=200,
                          num_layers=1, batch_first=True)


    def init_hidden(self, batch_size):
        h0 = torch.autograd.Variable(torch.zeros(1, batch_size, 200))
        c0 = torch.autograd.Variable(torch.zeros(1, batch_size, 200))
        return (h0, c0)


    def forward(self, x):
        batch_size = len(x)
        h0, c0 = self.init_hidden(batch_size)
        output, (h_n, c_n) = self.lstm(x, (h0, c0))
        return output

    # def __init__(self, embeddings, args):
    #     super(LSTM, self).__init__()
    #     self.args = args
    #     # self.hidden = self.init_hidden(20)
    #     self.lstm = nn.LSTM(input_size=200, hidden_size=100, num_layers=1, batch_first=True)

    # def init_hidden(self, batch_size):
    #     return (torch.autograd.Variable(torch.zeros(2,batch_size,100)),
    #             torch.autograd.Variable(torch.zeros(2,batch_size,100)))

class CNN(nn.Module):
    # def __init__(self, embeddings, args):
    #     super(CNN, self).__init__()
    #     self.args = args
    #     self.conv1 = nn.Conv1d(200, 200, kernel_size=3)


    # def forward(self, x):
    #     # x = x.permute(0,2,1)
    #     print x.size()
    #     x = x.unsqueeze(2)
    #     out = self.conv1(x)
    #     out = torch.mean(out, 2)
    #     # print("size of out", out.size())
    #     return out

class DomainClassifier(nn.Module):
    def __init__(self, embeddings, args):
        super(DomainClassifier, self).__init__()
        self.args = args
        self.seq = nn.Sequential(
                nn.Linear(100, 100),
                nn.ReLU(),
                nn.Linear(100,1))
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.seq(x)
        # print x.size()
        x = torch.squeeze(x)
        x = self.softmax(x)
        return x

def train(model, da_dan_model, train_data, max_epoches, dev_data, dev_labels, dev_data_android, dev_labels_android, domain_adaption=False, domain_adapation_batches=None, verbose=False):
    model.train()
    weight_decay = 1e-5# 1e-5
    lr = 1e-3# 1e-3
    dc_lr = 1e-3
    l = 1e-3 #lambda
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    domain_classifier_optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MultiMarginLoss(margin=0.2)
    criterion_da = nn.BCELoss()
    best_dev = 0.0
    corresponding_test = 0.0

    # dev_titles, dev_bodies = dev_data
    # dev_title_embs = Variable(torch.utils.data.TensorDataset(torch.FloatTensor(dev_titles)))
    # dev_body_embs = Variable(torch.utils.data.TensorDataset(torch.FloatTensor(dev_bodies)))
    # dev_title_embs = Variable(dev_titles)
    # dev_body_embs = Variable(dev_bodies)

    for epoch in range(max_epoches):
        print "epoch", epoch
        batch_num = 0
        for batch in train_data:
            # print batch_num
            batch_num+=1
            titles, bodies = batch
            title_embeddings = Variable(torch.FloatTensor(titles))
            # title_embeddings = Variable(titles)
            body_embeddings = Variable(torch.FloatTensor(bodies))
            # body_embeddings = Variable(bodies)
            title_output = model(title_embeddings)
            body_output = model(body_embeddings)
            # print "title input", np.array(titles).shape
            # print "title embeddings", title_embeddings.data.shape
            # print "title output shape", title_output.data.shape
            # question_embeddings = np.mean([title_output, body_output], axis=0)
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
                    # print i, j, i*20, i*20+j
                    # print type(question_embeddings[i*20])
                    if j != 0:
                        index = i * 20 + j
                        # print query_emb.size()
                        # print question_embeddings[index].size()
                        if query_emb.size()!=(100L,1L):
                            query_emb=torch.unsqueeze(query_emb, 1) 
                        # print query_emb.size()
                        question_embeddings_index = torch.unsqueeze(question_embeddings[index], 1) 
                        # print question_embeddings_index.size()
                        # X[i, j-1] = F.cosine_similarity(query_emb, question_embeddings[index], dim=1)
                        X.append(F.cosine_similarity(torch.t(query_emb), torch.t(question_embeddings_index), dim=1))

            # for i in range(20): # b rows, b = number of instances in a batch
            #     for j in range(21):
            #         X[i,j] = F.cosine_similarity(torch.FloatTensor(question_embeddings[i][0]), torch.FloatTensor(question_embeddings[i][j]))

            Y = np.array([0 for i in range(20)])
            
            optimizer.zero_grad()

            loss = criterion(torch.cat(X), Variable(torch.LongTensor(Y)))
            # print "loss", loss
            

            '''domain adaptation'''

            if domain_adaption:
                bodies = domain_adapation_batches[0]['bodies']
                titles = domain_adapation_batches[1]['titles']
                labels = domain_adapation_batches[0]['labels'] # should be same for bodies and titles
                labels = np.asarray(labels)

                da_train_batches = []
                for i in range(len(bodies)):
                    da_train_batches.append([bodies[i], titles[i]])
                da_model = DomainClassifier(da_train_batches, [])

                title_embeddings = Variable(torch.FloatTensor(titles))
                body_embeddings = Variable(torch.FloatTensor(bodies))
                title_output = da_dan_model(title_embeddings)
                body_output = da_dan_model(body_embeddings)
                question_embeddings = (title_output + body_output)/2.
                question_labels = da_model(question_embeddings)
                # print type(question_labels), type(labels)
                da_loss = criterion_da((question_labels), Variable(torch.FloatTensor(labels)))
                # print da_loss
                # print l*da_loss, loss
                total_loss = loss - l*da_loss
                domain_classifier_optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                domain_classifier_optimizer.step()

            else: # if not domain adaptation
                loss.backward()
                optimizer.step()

        if domain_adaption:
            evaluate(dev_data_android, dev_labels_android, da_dan_model)
        evaluate(dev_data, dev_labels, model) 

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

# train(model, train_batches, 50, dev_data, dev_labels)

train(DAN(train_batches, [200]), DAN(train_batches, [300]), train_batches, 50, dev_data, dev_labels, dev_data_android, dev_labels_android, domain_adaption=True, domain_adapation_batches=classifier_batches, verbose=False)
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

'''
dic: {q: ([p, p], [n, n, n, ...])}
for q:
    for p:
        create sample: (q, p, 20 randos)
'''

import numpy as np
from sklearn import metrics
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import gzip

torch.manual_seed(1)

batch_size = 20
hidden_dim = 300
weight_decay = 1e-5
lr = 1e-3

def read_file(fileName):
    text = []
    labels = []
    return text, labels

train_x, train_y = read_file("../data/askubuntu-master/train_random.txt")
dev_x, dev_y = read_file("../data/askubuntu-master/dev.txt")
test_x, test_y = read_file("../data/askubuntu-master/test.txt")

f = open('../data/askubuntu-master/vector/vectors_pruned.200.txt', 'r')
wv_text = [ ]
lines = f.readlines()
for line in lines:
    wv_text.append(line.strip())

word_to_vec = {}

for line in wv_text:
    parts = line.split()
    word = parts[0]
    vector = np.array([float(v) for v in parts[1:]])
    word_to_vec[word] = vector

def extract_features(data):
    features = [ ]
    for i in range(len(data)):
        num_words = 0
        current_feature = [ 0.0 for _ in range(200) ]
        for word in data[i].split():
            if word in word_to_vec:
                current_feature += word_to_vec[word]/np.linalg.norm(word_to_vec[word])
                num_words += 1

        if num_words > 0:
            current_feature /= num_words

        features.append(current_feature)

    return np.array(features)

train_x = extract_features(train_x)
dev_x = extract_features(dev_x)
test_x = extract_features(test_x)

train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(train_x), torch.LongTensor(train_y))
dev_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(dev_x), torch.LongTensor(dev_y))
test_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(test_x), torch.LongTensor(test_y))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
dev_loader = torch.utils.data.DataLoader(dev_dataset)
test_loader = torch.utils.data.DataLoader(test_dataset)

class DAN(nn.Module):

    def __init__(self, embeddings, args):
        super(DAN, self).__init__()
        self.args = args
        vocab_size, embedding_dim = embeddings.shape
        self.W_hidden = nn.Linear(embedding_dim, embedding_dim)
        self.W_out = nn.Linear(embedding_dim, 100)
        # self.seq = nn.Sequential(
        #         nn.Linear(input_dim, hidden_dim),
        #         nn.Tanh()              )

    def forward(self, x):
        all_embeddings = self.embedding_layer(x)
        avg_embeddings = torch.mean(all_embeddings, dim=1)
        hidden = nn.Tanh(self.W_hidden(avg_embeddings))
        return self.W_out(hidden)
        # x = self.seq(x)
        # return x


def evaluate(model, loader):
    model.eval()
    pred = []
    actual = []
    for data, label in loader:
        data, label = Variable(data), Variable(label)
        output = model(data)
        output = output.data.cpu().numpy()
        pred = np.concatenate((pred, np.argmax(output, axis=1)), axis=0)
        actual = np.concatenate((actual, label.data.cpu().numpy()), axis=0)

    return metrics.accuracy_score( y_pred=pred, y_true=actual)


def train(model, loader, max_epoches, dev_loader, test_loader, verbose=False):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.MultiMarginLoss()
    best_dev = 0.0
    corresponding_test = 0.0
    for epoch in range(max_epoches):
        for data in train_loader:
            data = Variable(data)
            title_embeddings = autograd.Variable(data['titles']) # janice pass this in
            body_embeddings = autograd.Variable(data['bodies'])
            title_output = model(title_embeddings)
            body_output = model(body_embeddings)
            question_output = np.mean(title_output, body_output, axis=0)
            '''
            create matrix by iterating from 0 to 20, 0 to 21:
            x = 20x21 matrix, mapping q to cosine similarity of each of 21 questions for each set of 22 questions
            y = list of positive question indices, which is always 0 in that row
            '''

            X = np.zeros((20,21))
            for i in range(20): # b rows, b = number of instances in a batch
                for j in range(21):
                    X[i,j] = F.cosine_similarity(question_embeddings[i][0], question_embeddings[i][j])

            Y = np.array([0 for i in range(20)])
            model.train()
            optimizer.zero_grad()

            loss = criterion(torch.cat(X), Variable(torch.FloatTensor(Y))
            loss.backward()
            optimizer.step()

        dev = evaluate(model, dev_loader)
        test = evaluate(model, test_loader)
        if dev > best_dev:
            best_dev = dev
            corresponding_test = test

    print (best_dev, corresponding_test)

model = FFN(300, hidden_dim, 2)

train(model, train_loader, 50, dev_loader, test_loader)

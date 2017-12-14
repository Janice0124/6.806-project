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
from tqdm import tqdm
import argparse

train_file = "../data/askubuntu-master/train_random.txt"
dev_file = "../data/askubuntu-master/dev.txt"
test_file = "../data/askubuntu-master/test.txt"
word_embs_file = "../data/askubuntu-master/vector/vectors_pruned.200.txt"
query_corpus_file = "../data/askubuntu-master/text_tokenized.txt.gz"

torch.manual_seed(1)

batch_size = 20

hidden_dim = 300
weight_decay = 1e-5
lr = 1e-3

# train_x, train_y = read_file("../data/askubuntu-master/train_random.txt")
# dev_x, dev_y = read_file("../data/askubuntu-master/dev.txt")
# test_x, test_y = read_file("../data/askubuntu-master/test.txt")
# f = open('../data/askubuntu-master/vector/vectors_pruned.200.txt', 'r')

# train_x = extract_features(train_x)
# dev_x = extract_features(dev_x)
# test_x = extract_features(test_x)

train_batches, dev_data, dev_labels, test_data, test_labels = inout.build_batches(train_file, dev_file, test_file, word_embs_file, query_corpus_file, 20)

# train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(train_x), torch.LongTensor(train_y))
# dev_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(dev_x), torch.LongTensor(dev_y))
# test_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(test_x), torch.LongTensor(test_y))

# train_loader = torch.utils.data.DataLoader(train_dataset)
# dev_loader = torch.utils.data.DataLoader(dev_dataset)
# test_loader = torch.utils.data.DataLoader(test_dataset)

class DAN(nn.Module):

    def __init__(self, embeddings, args):
        super(DAN, self).__init__()
        self.args = args
        vocab_size, embed_dim = embeddings.shape
        # self.embeddings = embeddings
        # embedding_dim = 440
        # self.W_hidden = nn.Linear(embedding_dim, embedding_dim)
        # self.W_out = nn.Linear(embedding_dim, 100)
        print len(embeddings)
        self.embedding_layer = nn.Embedding(vocab_size, embed_dim)
        self.embedding_layer.weight.data = torch.from_numpy(embeddings)
        self.embedding_layer.requires_grad = False
        self.W_hidden = nn.Linear(embed_dim, 200)
        self.W_out = nn.Linear(200, 100)
        # self.seq = nn.Sequential(
        #         nn.Linear(200, 100),
        #         nn.Tanh())

    def forward(self, x_indx):
        all_x = self.embedding_layer(x_indx)
        avg_x = torch.mean(all_x, dim=1)
        hidden = nn.Tanh(self.W_hidden(avg_x))
        # return self.seq(hidden)
        out = self.W_out(hidden)
        return out


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


def train(model, train_data, max_epoches, dev_data, dev_labels, args, verbose=False):
    if args.cuda:
        model = model.cuda()
    # model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.MultiMarginLoss()
    best_dev = 0.0
    corresponding_test = 0.0

    # dev_titles, dev_bodies = dev_data
    # dev_title_embs = Variable(torch.utils.data.TensorDataset(torch.FloatTensor(dev_titles)))
    # dev_body_embs = Variable(torch.utils.data.TensorDataset(torch.FloatTensor(dev_bodies)))
    # dev_title_embs = Variable(dev_titles)
    # dev_body_embs = Variable(dev_bodies)

    for epoch in range(max_epoches):
        data_loader = torch.utils.data.DataLoader(train_data, batch_size=20, shuffle=True, drop_last=True)
        model.train()
        for batch in tqdm(data_loader):
            sample_title_tensors, sample_body_tensors = torch.autograd.Variable(batch['sample_title'], torch.autograd.Variable(batch['sample_body']))
            candidate_title_tensors, candidate_body_tensors = torch.autograd.Variable(batch['candidate_title'], torch.autograd.Variable(batch['candidate_body']))
            # run through model titles first then bodies
            title_encodings = model(sample_title_tensors+candidate_title_tensors)
            body_encodings = model(sample_body_tensors+candidate_body_tensors)
            question_encodings = (title_encodings+body_encodings)/2.

            # if args.cuda:
            #     x, y = x.cuda(), y.cuda()

            X = []
            for i in range(20):
                for j in range(22):
                    print i, j, i*20, i*20+j, "i, j, i*20, i*20+j"
                    print question_embeddings[i*20].shape, "shape"
                    query_emb = torch.FloatTensor((question_embeddings[i * 20]))
                    if j != 0:
                        index = i * 20 + j
                        # prob want to reshape to 1 row
                        X[i, j-1] = F.cosine_similarity(query_emb, torch.FloatTensor(question_embeddings[index]), dim=1)

            # for i in range(20): # b rows, b = number of instances in a batch
            #         for j in range(21):
            #             X[i,j] = F.cosine_similarity(torch.FloatTensor(question_encodings[i][0]), torch.FloatTensor(question_encodings[i][j]))

            Y = np.array([0 for i in range(20)])
            X = torch.cat(X, 0), Y
            
            optimizer.zero_grad()

            loss = criterion(torch.cat(X), Variable(torch.FloatTensor(Y)))
            print loss
            loss.backward()
            optimizer.step()

        losses.append(loss.cpu().data[0])

    # Calculate epoch level scores
    avg_loss = np.mean(losses)
    return avg_loss

     # dev = evaluate(model, dev_loader)
     # test = evaluate(model, test_loader)
     # if dev > best_dev:
     #    best_dev = dev
     #    corresponding_test = test



        # for batch in train_data:
        #     titles, bodies = batch
        #     # title_embeddings = Variable(torch.FloatTensor(titles))
        #     title_embeddings = (torch.FloatTensor(titles))
        #     # title_embeddings = Variable(titles)
        #     # body_embeddings = Variable(torch.FloatTensor(bodies))
        #     body_embeddings = (torch.FloatTensor(bodies))
        #     # body_embeddings = Variable(bodies)
        #     title_output = model(title_embeddings)
        #     body_output = model(body_embeddings)
        #     question_embeddings = np.mean([title_output, body_output], axis=0)
        #     # len(question_embeddings) = 440 = 22 * 20
        #     '''
        #     create matrix by iterating from 0 to 20, 0 to 21:
        #     x = 20x21 matrix, mapping q to cosine similarity of each of 21 questions for each set of 22 questions
        #     y = list of positive question indices, which is always 0 in that row
        #     '''
        #     


            

        # dev_title_output = model(dev_title_embs)
        # dev_body_output = model(dev_body_embs)
        # dev_question_output = np.mean(dev_title_output, dev_body_output, axis=0)


    print (best_dev, corresponding_test)

embeddings, word_to_indx = inout.getEmbeddingTensor()
model = DAN(embeddings, [])

parser = argparse.ArgumentParser(description='Project')
parser.add_argument('--cuda', action='store_true', default=False, help='enable the gpu')
args = parser.parse_args()

train(model, train_batches, 50, dev_data, dev_labels, args)

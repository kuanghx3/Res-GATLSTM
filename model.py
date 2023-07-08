import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import copy
from tqdm import tqdm
from args import dev
import myfunctions as fn
import pandas as pd

class MultiHeadsGATLayer(nn.Module):
    def __init__(self, a_sparse, input_dim, out_dim, head_n, dropout=0, alpha=0.2):  # input_dim = seq_length
        super(MultiHeadsGATLayer, self).__init__()

        self.head_n = head_n
        self.heads_dict = dict()
        for n in range(head_n):
            self.heads_dict[n, 0] = nn.Parameter(torch.zeros(size=(input_dim, out_dim), device=dev))
            self.heads_dict[n, 1] = nn.Parameter(torch.zeros(size=(1, 2 * out_dim), device=dev))
            nn.init.xavier_normal_(self.heads_dict[n, 0], gain=1.414)
            nn.init.xavier_normal_(self.heads_dict[n, 1], gain=1.414)
        self.linear = nn.Linear(head_n, 1)

        # regularization
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=0)

        # sparse matrix
        self.a_sparse = a_sparse
        self.edges = a_sparse.indices()
        self.values = a_sparse.values()
        self.N = a_sparse.shape[0]
        a_dense = a_sparse.to_dense()
        a_dense[torch.where(a_dense == 0)] = -1000000000
        a_dense[torch.where(a_dense == 1)] = 0
        self.mask = a_dense

    def forward(self, x):
        b, n, s = x.shape
        x = x.reshape(b*n, s)

        atts_stack = []
        # multi-heads attention
        for n in range(self.head_n):
            h = torch.matmul(x, self.heads_dict[n, 0])
            edge_h = torch.cat((h[self.edges[0, :], :], h[self.edges[1, :], :]), dim=1).t()  # [Ni, Nj]
            atts = self.heads_dict[n, 1].mm(edge_h).squeeze()
            atts = self.leakyrelu(atts)
            atts_stack.append(atts)

        mt_atts = torch.stack(atts_stack, dim=1)
        mt_atts = self.linear(mt_atts)
        new_values = self.values * mt_atts.squeeze()
        atts_mat = torch.sparse_coo_tensor(self.edges, new_values)
        atts_mat = atts_mat.to_dense() + self.mask
        atts_mat = self.softmax(atts_mat)
        return atts_mat

class proposed_Model(nn.Module):
    def __init__(self, args, a_sparse):
        super(proposed_Model, self).__init__()
        self.seq_len = args.seq_len
        self.layer = 1
        self.gat_lyr = MultiHeadsGATLayer(a_sparse, self.seq_len, self.seq_len, head_n=1)
        self.gcn = nn.Linear(in_features=self.seq_len, out_features=self.seq_len)
        self.LSTM = nn.LSTM(input_size=self.layer, hidden_size=self.layer, num_layers=2, batch_first=True)
        self.MLP_decoder1 = torch.nn.Sequential(
            torch.nn.Linear(self.seq_len, int(args.MLP_hidden)),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(int(args.MLP_hidden), int(args.MLP_hidden / 2)),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(int(args.MLP_hidden / 2), 1))
        self.dropout = nn.Dropout(p=0.2)
        self.LeakyReLU = nn.LeakyReLU()
        self.alpha = args.alpha


    def forward(self, count):
        b, n, s = count.shape
        atts_mat = self.gat_lyr(count)  # dense(nodes, nodes)
        count_conv1 = torch.matmul(atts_mat, count)  # (b, n, s)
        count_conv1 = self.dropout(self.LeakyReLU(self.gcn(count_conv1)))
        count_conv1 = count_conv1 * (1-self.alpha) + count * self.alpha
        atts_mat = self.gat_lyr(count_conv1)  # dense(nodes, nodes)
        count_conv2 = torch.matmul(atts_mat, count_conv1)  # (b, n, s)
        count_conv2 = self.dropout(self.LeakyReLU(self.gcn(count_conv2)))
        count_conv2 = count_conv2 * (1 - self.alpha) + count_conv1 * self.alpha
        count_conv2 = count_conv2.reshape(b * n, s, self.layer)
        y,_ = self.LSTM(count_conv2)
        y = y.squeeze(2)
        y = self.MLP_decoder1(y)
        y = y.reshape(b,n)

        return y

def training(model, training_loader, args):
    optimizer = torch.optim.Adam(model.parameters())
    loss_function = torch.nn.MSELoss()
    for _ in tqdm(range(args.training_epochs)):
        for t, data in enumerate(training_loader):
            count, label = data
            count = count.permute(0, 2, 1)  # (batch, node, seq)
            optimizer.zero_grad()
            predict = model(count)
            loss = loss_function(predict, label)
            loss.backward()
            optimizer.step()

def test(model, test_loader, args):
    approach = "proposed"
    result = []
    model.eval()
    for t, data in enumerate(test_loader):
        count, label = data
        count = count.permute(0, 2, 1)  # (batch, node, seq)
        with torch.no_grad():
            predict = model(count)
        metrics = fn.get_metrics(predict.cpu().detach().numpy(), label.cpu().detach().numpy())
        result.append(metrics)
    pd.DataFrame(data=result, columns=["MSE", "RMSE", "MAPE", "RAE", "MAE", "R2"]).to_csv(
        "./result_" + str(args.predict_time) + "/" + approach + ".csv")
    return result


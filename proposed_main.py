import numpy as np
import pandas as pd
import torch
import os
import myfunctions as fn
from torch.utils.data import DataLoader
import model
from torch import nn
import scipy.sparse as sp
from args import args, dev
adj = pd.read_excel(args.data_path+"adj.xlsx", header=0, index_col=0)
adj_dense = np.array(adj, dtype=float)
adj_dense = torch.Tensor(adj_dense)
adj = adj_dense.to_sparse_coo().to(dev)

fn.seed_torch(2023)
training_numpy, test_numpy = fn.get_data(args)  # (batch, node)
training_dataset = fn.MyDataset(args, training_numpy, dev)  # count:(batch, LOOK_BACK, node)
testing_dataset = fn.MyDataset(args, test_numpy, dev)
training_loader = DataLoader(training_dataset, batch_size=len(training_dataset), shuffle=True, drop_last=True)
test_loader = DataLoader(testing_dataset, batch_size=len(testing_dataset), shuffle=False, drop_last=True)

model_proposed = model.proposed_Model(args, adj).to(dev)
model.training(model_proposed, training_loader, args)
torch.save(model_proposed, "./result_"+str(args.predict_time)+"/proposed.pt")
result = model.test(model_proposed, test_loader, args)
print(result)

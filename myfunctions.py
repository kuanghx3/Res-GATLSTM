import os
import torch
from torch import nn
import numpy as np
import random
import pandas as pd
from torch.utils.data import Dataset
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,mean_absolute_percentage_error

def creat_interval_dataset(dataset, lookback, predict_time):
    x = []
    y = []
    for i in range(len(dataset) - 2 * lookback):
        x.append(dataset[i:i + lookback])
        y.append(dataset[i + lookback + predict_time - 1])

    return np.array(x), np.array(y)


def seed_torch(seed):
    """
    Set all random seed
    Args:
        seed: random seed

    Returns: None

    """

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

def get_data(args):
    x = np.load(args.data_path+"data.npy") #(node, count)
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    n, c = x.shape
    data_training = x[:,:int(c*args.training_rate)]
    data_testing = x[:, int(c * args.training_rate):]
    return data_training.T, data_testing.T



def create_rnn_data(dataset, lookback, predict_time):
    x = []
    y = []
    for i in range(len(dataset) - lookback - predict_time):
        x.append(dataset[i:i + lookback, :])
        y.append(dataset[i + lookback + predict_time - 1, :])
    return np.array(x), np.array(y)

def create_rnn_data_t(dataset, lookback, predict_time):
    x = []
    y = []
    for day in range(dataset.shape[1]):
        for i in range(len(dataset) - lookback - predict_time):
            x.append(dataset[i:i + lookback, day])
            y.append(dataset[i + lookback + predict_time - 1, day])
    return np.array(x), np.array(y)

class MyDataset(Dataset):
    def __init__(self, args, data, dev):
        data, label = create_rnn_data(data, args.LOOK_BACK, args.predict_time)
        self.data = torch.Tensor(data)
        self.label = torch.Tensor(label)
        self.device = dev
        #print(self.temp.shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):  # occ: batch, seq, node
        return self.data[idx, :, :].to(self.device), self.label[idx, :].to(self.device)

def get_metrics(test_pre, test_real):

    MAPE = mean_absolute_percentage_error(test_real, test_pre)
    MAE = mean_absolute_error(test_real, test_pre)
    MSE = mean_squared_error(test_real, test_pre)
    RMSE = np.sqrt(MSE)
    R2 = r2_score(test_real, test_pre)
    RAE = np.sum(abs(test_pre - test_real)) / np.sum(abs(np.mean(test_real) - test_real))

    print('MAPE: {}'.format(MAPE))
    print('MAE:{}'.format(MAE))
    print('MSE:{}'.format(MSE))
    print('RMSE:{}'.format(RMSE))
    print('R2:{}'.format(R2))
    print(('RAE:{}'.format(RAE)))

    output_list = [MSE, RMSE, MAPE, RAE, MAE, R2]
    return output_list
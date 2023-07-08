import argparse
import warnings
import os
import torch
import sys
sys.path.append(os.getcwd())

from myfunctions import seed_torch

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument('--gpu', type=str, default='0', help='Select gpu device.')
parser.add_argument('--data_path', type=str,
                    default="./data/",
                    help='The directory containing the data.')
parser.add_argument('--LOOK_BACK', type=int, default=6,
                    help='Number of time step of the Look Back Mechanism.')
parser.add_argument('--predict_time', type=int, default=6,
                    help='Number of time step of the predict time.')
parser.add_argument('--nodes', type=int, default=58,
                    help='Number of parking lots.')
parser.add_argument('--training_epochs', type=int, default=2000,
                    help='The max training epochs.')
parser.add_argument('--seq_len', type=int, default=6,
                    help='Number of time step of the input sequence.')
parser.add_argument('--training_rate', type=float, default=0.7,
                    help='The rate of training set.')
parser.add_argument('--MLP_hidden', type=int, default=64,
                    help='Hidden size of MLP.')
parser.add_argument('--alpha', type=float, default=0.7,
                    help='Alpha.')
parser.add_argument('--layer', type=float, default=4,
                    help='number of HALSTM input feature.')


# args = parser.parse_args(args=[]) # jupyter
args = parser.parse_args()      # python

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
seed_torch(2023)

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
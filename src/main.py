import argparse
from ast import parse
import numpy as np
from load_data import load_data
# from attn_analysis import train
from train_batch import train

np.random.seed(555)


parser = argparse.ArgumentParser()


# amazon
parser.add_argument('--dataset', type=str, default='amazon', help='which dataset to use')
parser.add_argument('--n_epochs', type=int, default=50, help='the number of epochs')
parser.add_argument('--dim', type=int, default=32, help='dimension of user and entity embeddings')
parser.add_argument('--L', type=int, default=3, help='number of low layers')
parser.add_argument('--H', type=int, default=1, help='number of high layers')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--kge_batch_size', type=int, default=1024, help='kge batch size')
parser.add_argument('--l2_weight', type=float, default=1e-6, help='weight of l2 regularization')
# parser.add_argument('--l2_weight', type=float, default=1e-6, help='weight of l2 regularization')
parser.add_argument('--lr_rs', type=float, default=0.0002, help='learning rate of RS task')
parser.add_argument('--lr_kge', type=float, default=0.000001, help='learning rate of KGE task')
# parser.add_argument('--lr_kge', type=float, default=0.02, help='learning rate of KGE task')
parser.add_argument('--kge_interval', type=int, default=4, help='training interval of KGE task')
parser.add_argument('--TPS_DIR', type=str, default='../data/amazon/10-core/', help='which filtered dataset to use')
parser.add_argument('--n_entity', type=int, default=263878, help='the number of graph nodes, obtained from preprocess.py')
parser.add_argument('--num_heads', type=int, default=4, help='the number of GAT heads')
parser.add_argument('--update_attn', type=bool, default=True, help='wheather using attn update mechanism')
parser.add_argument('--adj_type', type=str, default='bi', help='Specify the type of the adjacency (laplacian) matrix from {bi, si}.')
show_loss = False
show_topk = False

args = parser.parse_args()

data = load_data(args)
train(args, data, show_loss, show_topk)



import random

import numpy as np
import torch
import argparse
from src.utils import *
from src import train


parser = argparse.ArgumentParser(description="Missing modality")

# Tasks
parser.add_argument(
    "--pretrained_model",
    type=str,
    default=None,
    help="name of the model to use (Transformer, etc.)",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="mosi",
    help="dataset to use (mosei, mosi, iemocap, sims)",
)
parser.add_argument(
    "--data_path",
    type=str,
    default=None,
    help="path for storing the dataset",
)

# Dropouts
parser.add_argument("--attn_dropout", type=float, default=0.1, help="attention dropout")
parser.add_argument(
    "--attn_dropout_a", type=float, default=0.1, help="attention dropout (for audio)"
)
parser.add_argument(
    "--attn_dropout_v", type=float, default=0.1, help="attention dropout (for visual)"
)
parser.add_argument("--relu_dropout", type=float, default=0.1, help="relu dropout")
parser.add_argument(
    "--embed_dropout", type=float, default=0.25, help="embedding dropout"
)
parser.add_argument(
    "--res_dropout", type=float, default=0.1, help="residual block dropout"
)
parser.add_argument(
    "--out_dropout", type=float, default=0.1, help="output layer dropout"
)

# Architecture
parser.add_argument(
    "--nlevels", type=int, default=5, help="number of layers in the network"
)
parser.add_argument(
    "--proj_dim", type=int, default=30, help="projection dimension of the network"
)
parser.add_argument(
    "--num_heads",
    type=int,
    default=5,
    help="number of heads for the transformer network",
)
parser.add_argument(
    "--attn_mask", action="store_false", help="use attention mask for Transformer"
)
parser.add_argument("--prompt_dim", type=int, default=30)
parser.add_argument("--prompt_length", type=int, default=16)


# Tuning
parser.add_argument(
    "--batch_size", type=int, default=64, metavar="N", help="batch size"
)
parser.add_argument("--clip", type=float, default=0.8, help="gradient clip value")
parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
parser.add_argument("--optim", type=str, default="Adam", help="optimizer to use")
parser.add_argument("--num_epochs", type=int, default=40, help="number of epochs")
parser.add_argument("--when", type=int, default=10, help="when to decay learning rate")
parser.add_argument("--drop_rate", type=float, default=0.6)


# Logistics
parser.add_argument(
    "--log_interval",
    type=int,
    default=30,
    help="frequency of result logging (default: 30)",
)
parser.add_argument("--seed", type=int, default=666, help="random seed")
parser.add_argument("--no_cuda", action="store_true", help="do not use cuda")
parser.add_argument("--name", type=str, default=None, help="name of the trial")
args = parser.parse_args()


dataset = str.lower(args.dataset.strip())

output_dim_dict = {"mosi": 1, "mosei": 1, "sims": 1, "iemocap": 4}

criterion_dict = {"iemocap": "CrossEntropyLoss"}


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


torch.set_default_tensor_type("torch.FloatTensor")
use_cuda = False
if torch.cuda.is_available():
    if args.no_cuda:
        print(
            "WARNING: You have a CUDA device, so you should probably not run with --no_cuda"
        )
    else:
        torch.cuda.manual_seed(args.seed)
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        use_cuda = True

setup_seed(args.seed)

####################################################################
#
# Load the dataset
#
####################################################################

dataloaders, orig_dims, n_nums, seq_len = get_loader(args)
trainloder = dataloaders["train"]
validloder = dataloaders["valid"]
testloder = dataloaders["test"]
####################################################################
#
# Hyperparameters
#
####################################################################
hyp_params = args
hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = orig_dims
hyp_params.layers = args.nlevels
hyp_params.use_cuda = use_cuda
hyp_params.dataset = dataset
hyp_params.when = args.when
hyp_params.n_train, hyp_params.n_valid, hyp_params.n_test = n_nums
hyp_params.output_dim = output_dim_dict.get(dataset, 1)
hyp_params.criterion = criterion_dict.get(dataset, "L1Loss")
hyp_params.seq_len = seq_len


if __name__ == "__main__":
    train.initiate(hyp_params, trainloder, validloder, testloder)

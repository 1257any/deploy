# %%
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,ConcatDataset
import os,sys,time

from model_MLP_regressor import *
from utils import *

## check available device
device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
print("device:", device)

init_random_seed(random_seed=42)

##================================================================================================
project = "NCI"

target_type = "beta"
print("target_type:", target_type)

##-------------------------------------------------
n_inputs = 512
n_hiddens = 512
dropout = 0.2
batch_size = 32
learning_rate = 0.0001           
print("dropout:", dropout)
print("batch_size:", batch_size)
print("learning_rate:", learning_rate)

max_epochs,patience = 500,50

# +
try:
    ik_fold = int(sys.argv[1])
    il_fold = int(sys.argv[2])
    i_gene_min = int(sys.argv[3])
    i_gene_step = int(sys.argv[4])
except:
    ik_fold = 0
    il_fold = 0
    i_gene_min = 0
    i_gene_step = 10

print("project:", project)
print("ik_fold:", ik_fold)
print("il_fold:", il_fold)

# -

print("ik_fold:", ik_fold)
print("il_fold:", il_fold)
print("i_gene_min:", i_gene_min)
print("i_gene_step:", i_gene_step)
print("max_epochs: {}, patience: {}".format(max_epochs, patience))

path2features = f"../auto_encoder/"
path2target = "../metadata/"
path2split = "../metadata/"

gene_file = f"{project}_sites.csv"
print("gene_file:", gene_file)

i_gene_max = int(i_gene_min + i_gene_step)

genes = pd.read_csv(gene_file)
genes = genes["sites"].values
genes = genes[i_gene_min:i_gene_max]

print("genes:", genes)
print("len(genes):", len(genes))

## create result directory
result_dir = f"results/result_{ik_fold}_{il_fold}/"
os.makedirs(result_dir,exist_ok=True)


##================================================================================================
train_set, valid_set, test_set = load_dataset(path2features, path2target, path2split, target_type, \
                                              ik_fold, il_fold, genes, project)

bias_init = torch.nn.Parameter(torch.Tensor(np.mean([sample[1].detach().cpu().numpy()\
                         for sample in train_set], axis=0)).to(device))
n_outputs = len(genes)

model = MLP_regression(n_inputs, n_hiddens, n_outputs, dropout, bias_init)
model.to(device)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

##================================================================================================
print(" ")
print(" --- fit --- ")

model,train_loss,train_coef,train_slope,\
valid_loss,valid_coef,valid_slope,valid_labels,valid_preds = \
fit(model, optimizer, train_set, valid_set, max_epochs, patience, batch_size)

analyze_result(result_dir,genes,model,train_loss,train_coef,train_slope, \
               valid_loss,valid_coef,valid_slope,valid_labels, valid_preds, test_set)

print("--- completed ---")

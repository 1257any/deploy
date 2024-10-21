# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os,sys,time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,precision_score,recall_score

import torch
from torch import nn
from torch.utils.data import DataLoader,Subset,Dataset
from model_multiclass_classifier import *
from utils_multiclass_classifier import *
from scipy import stats
import joblib

## check available device
device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
print("device:", device)

init_random_seed(random_seed=42)

# %%
##==============================================================================================
## inputs:
project = "NCI"
extension = "_10class"

feature_rescale = True
class_weights_based_tile = True

try:
    ik_fold = int(sys.argv[1])
    il_fold = int(sys.argv[2])
except:
    ik_fold = 0
    il_fold = 0

print("project:", project)
print("ik_fold:", ik_fold)
print("il_fold:", il_fold)

path2features = f"../auto_encoder/"
path2target = "../metadata/"
path2split = "../metadata/"

feature_file = f"{project}_features_AE{extension}.npy"
target_file = f"{project}_slide_selected{extension}.csv"
split_file = f"{project}_train_valid_test_idx{extension}.npz"
target_col = "DBTA_class_idx"

## hyper-parameters:
max_epochs,patience = 300,30
#max_epochs,patience = 3,3

n_inputs = 512
n_hiddens = 512
#n_hiddens = 16

weight_factor = 0.2
dropout = 0.2
batch_size = int(1024*64)
learning_rate = 0.0001

## create result directory
result_dir = f"results/result_{ik_fold}_{il_fold}/"
os.makedirs(result_dir,exist_ok=True)

# %%
##==============================================================================================
## load data
tile_features, tile_target, n_tiles, cumsum, slide_target = load_data(path2features, path2target,\
                                                feature_file, target_file, target_col)

print("tile_features.shape:", tile_features.shape)  ##[n_tiles_total,512]
print("tile_target.shape:", tile_target.shape)      ##[n_titles_total]
#print("n_tiles:", n_tiles)

n_slides = cumsum.shape[0] - 1
n_outputs = len(np.unique(slide_target))

print(f"n_slides: {n_slides}, n_classes: {n_outputs}")

##------------------------------------------------------
## load_train_valid_test_idx (slide level):
train_valid_test_idx = np.load(f"{path2split}{split_file}", allow_pickle=True)

train_idx = train_valid_test_idx["train_idx"][ik_fold][il_fold]
valid_idx = train_valid_test_idx["valid_idx"][ik_fold][il_fold]
test_idx = train_valid_test_idx["test_idx"][ik_fold]

# %%
##------------------------------------------------------
## get features and target at tile level
train_features, train_target = get_features_target(tile_features, tile_target, cumsum, train_idx)
valid_features, valid_target = get_features_target(tile_features, tile_target, cumsum, valid_idx)
test_features, test_target = get_features_target(tile_features, tile_target, cumsum, test_idx)

print(train_features.shape, train_target.shape)
print(valid_features.shape, valid_target.shape)
print(test_features.shape, test_target.shape)

# %%
####==============================================================================================
## model
model = MLP_classifier(n_inputs, n_hiddens, n_outputs, dropout)
model.to(device)
#print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


if class_weights_based_tile:
    class_weights = sk_class_weights(train_target)
    print("class_weights_based_tile:", class_weights)
    np.savetxt(f"{result_dir}tile_weights.txt", class_weights, fmt="%s")

else:
    class_weights = sk_class_weights(slide_target[train_idx])
    print("class_weights_based_slide:", class_weights)
    np.savetxt(f"{result_dir}slide_weights.txt", class_weights, fmt="%s")

loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights).to(device))

##----------------------------------------
## rescale
if feature_rescale:
    sc = StandardScaler()
    #sc = MinMaxScaler()
    train_features = sc.fit_transform(train_features)
    valid_features = sc.transform(valid_features)
    test_features = sc.transform(test_features)

    ## save trained scaler:
    joblib.dump(sc, f"{result_dir}sc_trained.joblib")
##------------------

train_set = dataset(train_features, train_target)
valid_set = dataset(valid_features, valid_target)
test_set = dataset(test_features, test_target)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
#unshuffled_train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=0)

# %%
##----------------------------------------
print(" --- fit --- ")
start_time = time.time()

## model fitting
model,train_loss,valid_loss,valid_auc,valid_ap,slide_valid_auc,slide_valid_ap = \
fit(model, optimizer, loss_fn, max_epochs, patience, device,\
    n_tiles, train_loader, valid_loader, valid_idx)

# %%
##----------------------------------------
## plot result
plot_result(result_dir,model,train_loss,valid_loss,valid_auc,valid_ap,slide_valid_auc,slide_valid_ap)

# %%
####==============================================================================================
## predict on the entire test set
_, test_label, test_prob = predict(model, test_loader, loss_fn, device)

# %%
## slide label 
slide_test_label = tile2slide(test_label, n_tiles, test_idx)
slide_test_prob = tile2slide(test_prob, n_tiles, test_idx)
slide_test_pred = np.argmax(slide_test_prob,axis=1)

## evaluation at slide level:
test_auc = roc_auc_score(slide_test_label, slide_test_prob, average="macro", multi_class="ovr")
test_ap = precision_score(slide_test_label, slide_test_pred, average="micro",zero_division=0)
#test_recall = recall_score(slide_test_label, slide_test_pred, average="micro")

print(f"test_auc: {round(test_auc,4)}, test_ap: {round(test_ap,4)}")

np.savetxt(f"{result_dir}label_pred.txt", np.hstack((np.array((slide_test_label, slide_test_pred)).T, slide_test_prob)), fmt="%s")
np.savetxt(f"{result_dir}auc_ap.txt", np.array((round(test_auc,4),round(test_ap,4))), fmt="%s")

# %%
####==============================================================================================
print(f"fit -- completed --: {round(time.time() - start_time, 2)}s")

# %%

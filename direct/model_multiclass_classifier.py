#!/usr/bin/env python
# coding: utf-8
import numpy as np
import torch
from torch import nn
from scipy.special import expit,softmax
#from tqdm import tqdm
#import time
from utils_multiclass_classifier import *

from sklearn.metrics import roc_auc_score,average_precision_score

##================================================================================================
class MLP_classifier(nn.Module):
    def __init__(self,n_inputs,n_hiddens,n_outputs, dropout):
        super(MLP_classifier,self).__init__()

        self.layer0 = nn.Sequential(nn.Linear(n_inputs, n_hiddens, bias=True),
            #nn.ReLU(),
            nn.Dropout(dropout)
            )

        self.layer1 = nn.Sequential(nn.Linear(n_hiddens, n_outputs, bias=True),
            #nn.ReLU()
            )
    
    ##-----------------------------------
    def forward(self,x):
        x = self.layer0(x)
        x = self.layer1(x)
        #x = torch.sigmoid(x)
        #x = torch.softmax(x,axis=1)

        return x

##================================================================================================
def training_epoch(model, optimizer, train_loader, loss_fn, device):
    model.train()

    loss_list = []
    for x,y in train_loader:       ## repeat times = len(train_set)/batch_size

        pred = model(x.float().to(device))       

        loss = loss_fn(pred, y.long().to(device))

        loss_list.append(loss.item()) ## add loss of each sample to a list

        ## reset gradients to zero
        optimizer.zero_grad()

        ## compute gradients
        loss.backward()

        ## update parameters using gradients
        optimizer.step()
    
    return np.mean(loss_list) #, labels, preds

##================================================================================================
def predict(model, dataloader, loss_fn, device):
    model.eval()
    
    labels = []
    preds = []
    loss_list = []
    with torch.no_grad():
        for x, y in dataloader:         ## load x, y from dataloader for each batch
            
            labels.append(y.numpy())  ## y_actual

            pred = model(x.float().to(device))                ## y_pred = model(x)

            loss_list.append(loss_fn(pred, y.long().to(device)).item())

            preds.append(pred.detach().cpu().numpy())
    
    ## convert list to 2D array
    labels = np.concatenate(labels)
    preds = np.concatenate(preds)

    ## convert logit to probs
    preds = softmax(preds,1)
    #print("preds.shape:", preds.shape)

    return np.mean(loss_list),labels,preds

##================================================================================================
def fit(model, optimizer, loss_fn, max_epochs, patience, device,\
    n_tiles, train_loader, valid_loader, valid_idx):
        #n_tiles, train_loader, unshuffled_train_loader, train_idx, valid_loader, valid_idx):
        #test_loader, test_idx):

    train_loss = [] ; train_auc = [] ; train_ap = [] ; slide_train_auc = [] ; slide_train_ap = []
    valid_loss = [] ; valid_auc = [] ; valid_ap = [] ; slide_valid_auc = []; slide_valid_ap = []
    #test_loss = [] ; test_auc = [] ; test_ap = [] ; slide_test_auc = []; slide_test_ap = []

    epoch_since_best = 0
    valid_auc_old = -1. ; valid_ap_old = -1. ; slide_valid_ap_old = -1.
    for e in range(max_epochs):
        epoch_since_best += 1

        ## train
        train_loss1 = training_epoch(model, optimizer, train_loader, loss_fn, device)

        ## predict on valid set
        valid_loss1, valid_label, valid_prob = predict(model, valid_loader, loss_fn, device)

        valid_pred = np.argmax(valid_prob,axis=1)

        ##--------------------------
        slide_valid_label,slide_valid_prob = tile2slide_label_prob(valid_label,valid_prob,n_tiles,valid_idx)
        slide_valid_pred = np.argmax(slide_valid_prob,axis=1)

        valid_auc1 = roc_auc_score(valid_label, valid_prob, average="macro", multi_class="ovr")
        valid_ap1 = precision_score(valid_label, valid_pred, average="micro",zero_division=0)
        slide_valid_auc1 = roc_auc_score(slide_valid_label, slide_valid_prob, average="macro", multi_class="ovr")
        slide_valid_ap1 = precision_score(slide_valid_label, slide_valid_pred, average="micro",zero_division=0)
        
        ##--------------------------
        train_loss.append(train_loss1)

        valid_loss.append(valid_loss1)
        valid_auc.append(valid_auc1) ; valid_ap.append(valid_ap1)
        slide_valid_auc.append(slide_valid_auc1) ; slide_valid_ap.append(slide_valid_ap1)

  
        if valid_ap1 > valid_ap_old:
            epoch_since_best = 0
            valid_ap_old = valid_ap1

        if epoch_since_best == patience:
            print('Early stopping at epoch {}'.format(e + 1))
            break
        
    return model,train_loss,valid_loss,valid_auc,valid_ap,slide_valid_auc,slide_valid_ap

##================================================================================================
##================================================================================================
def plot_result(result_dir,model,train_loss,valid_loss,valid_auc,valid_ap,slide_valid_auc,slide_valid_ap):
    #test_loss,test_auc,test_ap,slide_test_auc,slide_test_ap):

    ## save trained model
    torch.save(model.state_dict(), f"{result_dir}model_trained.pth")

    nx,ny = 5,1
    fig, ax = plt.subplots(ny,nx,figsize=(nx*4,ny*3))

    ax[0].plot(train_loss, 'k-', label="train")
    ax[0].plot(valid_loss, 'b--', label="valid")

    ax[1].plot(valid_auc, 'b--', label="valid")
    ax[2].plot(valid_ap, 'b--', label="valid")
    ax[3].plot(slide_valid_auc, 'b--', label="valid")
    ax[4].plot(slide_valid_ap, 'b--', label="valid")

    ax[1].set_title(f"tile valid AUC: {round(valid_auc[-1],4)}")
    ax[2].set_title(f" tile valid AP: {round(valid_ap[-1],4)}")

    ax[3].set_title(f"slide valid AUC: {round(slide_valid_auc[-1],4)}")
    ax[4].set_title(f"slide valid AP: {round(slide_valid_ap[-1],4)}")

    for i in range(nx):
        ax[i].set_xlabel("n_epochs")
        ax[i].legend()

    ax[0].set_ylabel("loss")
    ax[1].set_ylabel("tile-level AUC")
    ax[2].set_ylabel("tile-level AP")
    ax[3].set_ylabel("slide-level AUC")
    ax[4].set_ylabel("slide-level AP")

    plt.tight_layout(h_pad=1, w_pad= 1.0)
    plt.savefig(f"{result_dir}loss.pdf", format='pdf', dpi=50)
##================================================================================================




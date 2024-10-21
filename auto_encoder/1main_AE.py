import torch
from torch.utils.data import DataLoader
import time,sys
import numpy as np
from sklearn.model_selection import train_test_split
from model_AE import *
from torch.utils.data import Subset,Dataset

## check available device
device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
print("device:", device)

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
init_random_seed(random_seed=42)
##================================================================================================
##================================================================================================

project = "NCI"

path2features = f"../slide_processing/"
path2outputs = ""

##========================================
## hyper parameters:
max_epochs = 500

n_inputs = 2048
n_hiddens = 512
n_outputs = 2048
batch_size = 32
lr = 0.0001

save_each_project_to_file = True
analyze_features_hist = True
##================================================================================================
## load data
features = np.load(f"{path2features}{project}_features.npy", allow_pickle=True)
    
##-------------
n_slides = len(features)
print(f"n_slides: {n_slides}")

train_idx, test_idx = train_test_split(np.arange(n_slides), test_size=0.1, shuffle=True)
print(f"train_idx.shape: {train_idx.shape}, test_idx.shape: {test_idx.shape}")

dataset = SlideDataset(features)
train_set = Subset(dataset, train_idx)
test_set = Subset(dataset, test_idx)

print(f"combined len(train_set): {len(train_set)}, len(test_set): {len(test_set)}")

##================================================================================================
model = AutoEncoder(n_inputs, n_hiddens, n_outputs)
model.to(device)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

model,train_loss,test_loss = fit(model, optimizer, train_set, test_set, max_epochs, batch_size, device)

##-------------------------
## save model and plot loss:
torch.save(model.state_dict(), f"{project}_model_AE.pth")
np.savetxt("loss.txt", np.array((train_loss, test_loss)).T, fmt="% f")

nx,ny = 1,1
fig, ax = plt.subplots(ny,nx,figsize=(nx*4,ny*3))

ax.plot(train_loss,"o--", label="train")
ax.plot(test_loss,"^--", label="test")

ax.set_xlabel("n_epochs")
ax.set_ylabel("loss")
ax.legend()

plt.tight_layout(h_pad=1, w_pad= 1.0)
plt.savefig(f"loss_{project}.pdf", format="pdf", dpi=50)

##================================================================================================
## Compress entire data
print(" ")
print("--- compress data, with the same order ---")
    
features_AE = features_compression(model,path2features,project,device)

if save_each_project_to_file:
    np.save(f"{path2outputs}{project}_features_AE.npy", features_AE1)

print("--- completed --- feature compression")

##================================================================================================
if analyze_features_hist:
    print(" ")
    start_time = time.time()
    analyze_result(features,features_AE)
    print(f"analyze time: {int(time.time() - start_time)}s")
    print("--- completed --- analyze_result")
##================================================================================================

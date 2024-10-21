import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,precision_score,\
recall_score,f1_score,roc_curve,cohen_kappa_score,matthews_corrcoef,classification_report

import seaborn as sns
##===================================================================================================
def load_data(path2features,path2target,feature_file, target_file, target_col):
    ##-----------------------------------
    ## load features
    features_list = np.load(f"{path2features}{feature_file}", allow_pickle=True)

    n_slides = len(features_list)
    #print("n_slides:", n_slides)

    n_tiles = np.zeros(n_slides).astype(int)
    features_tile = []
    for i_slide in range(n_slides):

        features = features_list[i_slide][1]
        n_tiles[i_slide] = features.shape[0]

        features_tile.append(features)

    features_tile = np.concatenate(features_tile)

    n_tiles_total = n_tiles.sum()
    print("n_tiles_total:", n_tiles_total)

    cumsum = np.cumsum(n_tiles, dtype=int)
    cumsum = np.hstack(([0], cumsum))
    #print("cumsum:", cumsum) ## [0, 647, 1153, 1560, ..., 25540]

    ##-----------------------------------
    ## load target
    df = pd.read_csv(f"{path2target}{target_file}")

    #target = df["patient_response"].values
    target = df[target_col].values
    #print("target:", target)
    #print("target.shape:", target.shape)

    #target = label_encoder.fit_transform(target)

    target_tile = np.zeros(n_tiles_total, dtype=int)
    for i_slide in range(n_slides):
        target_tile[cumsum[i_slide]:cumsum[i_slide+1]] = target[i_slide]
    
    return features_tile, target_tile, n_tiles, cumsum, target

##===================================================================================================
def select_balanced_idx(train_idx, slide_target, balanced_seed):
    ## note: 'slide_target' should be {0,1}
    
    np.random.seed(balanced_seed)

    n1 = (slide_target == 1).sum()
    n0 = (slide_target == 0).sum()
    
    if n1 < n0:
        train_idx1 = np.array([i for i in train_idx if slide_target[i] == 1])
        train_idx0 = np.setdiff1d(train_idx, train_idx1)
        selected_train_idx0 = np.random.choice(train_idx0, len(train_idx1), replace=False)
        balanced_train_idx = np.hstack((train_idx1, selected_train_idx0))        
    else:
        train_idx0 = np.array([i for i in train_idx if slide_target[i] == 0])
        train_idx1 = np.setdiff1d(train_idx, train_idx0)
        selected_train_idx1 = np.random.choice(train_idx1, len(train_idx0), replace=False)
        balanced_train_idx = np.hstack((selected_train_idx1, train_idx0))  
    
    return balanced_train_idx
##===================================================================================================
def tile2slide(tile_level, n_tiles, slide_idx):
    ## convert tile level to slide level
    n_tiles = n_tiles[slide_idx]
    cumsum = np.cumsum(n_tiles, dtype=int)  ## cumsum of test slides
    cumsum = np.hstack(([0], cumsum))       ##

    #print("n_tiles:", n_tiles)
    #print("cumsum:", cumsum)

    slide_level_mean = np.array([np.mean(tile_level[cumsum[i]:cumsum[i+1]],axis=0) for i in range(len(slide_idx))])
    #slide_level_median = np.array([np.median(tile_level[cumsum[i]:cumsum[i+1]]) for i in range(len(slide_idx))])

    return slide_level_mean#, slide_level_median

##------------------
def tile2slide_label_prob(tile_level_label, tile_level_prob, n_tiles, slide_idx):
    ## convert tile level to slide level
    n_tiles = n_tiles[slide_idx]
    cumsum = np.cumsum(n_tiles, dtype=int)  ## cumsum of test slides
    cumsum = np.hstack(([0], cumsum))       ##

    #print("n_tiles:", n_tiles)
    #print("cumsum:", cumsum)

    slide_level_label = np.array([np.mean(tile_level_label[cumsum[i]:cumsum[i+1]],axis=0) for i in range(len(slide_idx))])

    slide_level_prob = np.array([np.mean(tile_level_prob[cumsum[i]:cumsum[i+1]],axis=0) for i in range(len(slide_idx))])

    #slide_level_median = np.array([np.median(tile_level[cumsum[i]:cumsum[i+1]]) for i in range(len(slide_idx))])

    return slide_level_label, slide_level_prob#, slide_level_median

##===================================================================================================
def get_features_target(features_tile, target_tile, cumsum, slide_idx):
## get features and target at tile level from slide_idx
    features = []
    target = []
    for i in slide_idx:
        features.append(features_tile[cumsum[i]:cumsum[i+1]])
        target.append(target_tile[cumsum[i]:cumsum[i+1]])
    
    features = np.concatenate(features)
    target = np.concatenate(target)
            
    return features,target

##===================================================================================================
class dataset(Dataset):
    def __init__(self,x,y):
        self.x = torch.tensor(x,dtype=torch.float32)
        self.y = torch.tensor(y,dtype=torch.float32)
        self.length = self.x.shape[0]

    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]

    def __len__(self):
        return self.length

##===================================================================================================
def find_topk_pred_names(actual_names, pred_scores, class_names, topk):
    
    ##-----------------
    ## increased sort
    i = np.argsort(pred_scores,axis=1) 
    
    ## decreased sort
    i1 = np.flip(i, axis=1)
    
    ## select top k
    top_idx = i1[:, :topk]
    
    #return top_idx
    all_topk_pred_names = class_names[top_idx]
    
    ##-----------------
    ## assign pred to actual if actual is in topk, otherwise top1
    n_slides = len(actual_names)
    topk_pred_names = []
    for i_slide in range(n_slides):
        if actual_names[i_slide] in all_topk_pred_names[i_slide,:]:
            topk_pred_names.append(actual_names[i_slide])
        else:
            topk_pred_names.append(all_topk_pred_names[i_slide,0])
            
    return topk_pred_names, all_topk_pred_names

##===================================================================================================
def plot_cfs(df_cfs, acc, sensitivity, specificity, fig_name):
    nx,ny = 1,1
    fig, ax = plt.subplots(ny,nx,figsize=(nx*30,ny*30))

    fontsize=12

    ##----------------------------------
    ## confusion matrix
    fontsize=14
    heatmap = sns.heatmap(df_cfs, ax=ax, cmap='Blues', fmt='d', annot=True, annot_kws={"size": fontsize+4},
              linewidths=2, linecolor='black', cbar=False)

    ax.set_xlabel('actual class', fontsize=fontsize, color='black')
    ax.set_ylabel('predicted class', fontsize=fontsize, color='black')
    ax.set_title(f"acc: {round(acc,3)}, sensitivity: {round(sensitivity,3)}, specificity: {round(specificity,3)}",  pad=15, fontsize=fontsize+8, color='black')

    plt.tight_layout(h_pad=1, w_pad= 1.5)
    plt.savefig(f"{fig_name}.pdf", format='pdf', dpi=50)
    plt.close()

##===================================================================================================
def sk_class_weights(labels):
    u, c = np.unique(labels, return_counts=True)
    n_samples = sum(c)

    weights = n_samples/(len(c)*c)
    
    return weights

##===================================================================================================
def model_performance(actual, pred):
    
    class_names = np.union1d(actual, pred)
    
    cfs_matrix = confusion_matrix(actual, pred, labels=class_names)
    df_cfs = pd.DataFrame(cfs_matrix, index=class_names, columns=class_names)
    
    ##-----------------
    report = classification_report(actual, pred, labels=class_names, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report).transpose()

    precision = df_report.loc[class_names]["precision"]
    recall = df_report.loc[class_names]["recall"]
    
    micro_precision = df_report.loc["accuracy"]["precision"]
    weighted_precision = df_report.loc["weighted avg"]["precision"]
    
    #print(f"micro_precision: {micro_precision}, weighted_precision: {weighted_precision}")

    ## other performance metrics:
    kappa_score = cohen_kappa_score(actual, pred)
    mat_corrcoef = matthews_corrcoef(actual, pred)
    #print(f"kappa_score: {kappa_score}, matthews_corrcoef: {mat_corrcoef}")
    
    ##-----------------
    TP = np.diag(cfs_matrix)
    FN = np.sum(cfs_matrix,axis=1) - TP
    FP = np.sum(cfs_matrix,axis=0) - TP
    TN = np.sum(cfs_matrix) - (TP + FN + FP)

    ## averall accuracy
    ACC = (TP+TN)/(TP+FN+FP+TN)

    ## Precision
    PPV = np.divide(TP, (TP+FP), out=np.zeros_like(TP, dtype=float), where=(TP+FP)!=0)

    ## Recall (sensitivity)
    RC = np.divide(TP, (TP+FN), out=np.zeros_like(TP, dtype=float), where=(TP+FN)!=0)

    ## micro-precision
    precision_overall = sum(TP)/sum(TP+FP)  

    ## micro-recall
    micro_recall = sum(TP)/sum(TP+FN)  

    ## error rate
    error_rate = FN/np.sum(cfs_matrix,axis=1)
    #print("error_rate:", error_rate)
    cumsum_cfs = np.cumsum(cfs_matrix,axis=1)

    #print(f"acc: {ACC}, precision: {PPV}, recall: {recall}")

    return micro_precision, micro_recall, precision, recall, \
            error_rate, cumsum_cfs, df_cfs, df_report
##===================================================================================================
def init_random_seed(random_seed=42):
    # Python RNG
    np.random.seed(random_seed)

    # Torch RNG
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

##===================================================================================================




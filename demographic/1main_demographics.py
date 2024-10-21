# %%
import numpy as np
import pandas as pd
import os,sys,time
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.preprocessing import MinMaxScaler,StandardScaler
#from sklearn.calibration import calibration_curve

from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,\
recall_score,roc_curve,auc,cohen_kappa_score,matthews_corrcoef,classification_report

from model_multiclass import *
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import OneHotEncoder
import joblib

np.random.seed(42)

# %%
##====================================================================================
project = "NCI"
save_trained_model = True
extension = "_10class"

try:
    model_name = sys.argv[1]
except:
    model_name = "LR"

result_dir = f"results_{model_name}/"
path2meta = "../metadata/"

os.makedirs(result_dir,exist_ok=True)

# %%
##====================================================================================
## load data
df_data = pd.read_csv(f"{path2meta}{project}_slide_selected{extension}.csv")
df_dict = pd.read_csv(f"{path2meta}NCI_dict{extension}.csv")

class_names = df_dict["DBTA_class"].values
actual_names = df_data["DBTA_class"].values
actual_idxs = np.array([np.argwhere(class_names == x)[0][0] for x in actual_names])

y = actual_idxs

X_cont = df_data["age"].values
X2 = df_data["location"].values
X3 = df_data["sex"].values

X2_onehot = OneHotEncoder().fit_transform(X2.reshape((-1,1))).toarray()
joblib.dump(onehot, "location_trained_onehot.joblib")

X3 = (X3 == "M").astype(int)
X_onehot = np.hstack([X2_onehot, X3.reshape(-1,1)])

# %%
##====================================================================================
## model
model, hyper_parameters = multi_class(model_name)

# Create grid search using cross validation
gcv = GridSearchCV(model, hyper_parameters, 
                   scoring='f1_micro',
                   #scoring='ovr',
                   cv=5, n_jobs=-1)

# %%
##====================================================================================
## patient split
train_valid_test_idx = np.load(f"{path2meta}{project}_train_valid_test_idx{super_class}.npz", allow_pickle=True)

sc = MinMaxScaler()
#sc = StandardScaler()

#ik_fold = 0
il_fold = 0
for iik, ik_fold in enumerate(np.arange(5)):
    print("ik_fold:", ik_fold)

    train_idx = train_valid_test_idx["train_idx"][ik_fold][il_fold]
    valid_idx = train_valid_test_idx["valid_idx"][ik_fold][il_fold]
    test_idx = train_valid_test_idx["test_idx"][ik_fold]

    train_idx = np.hstack((train_idx, valid_idx))

    X_cont_train, X_onehot_train, y_train = X_cont[train_idx], X_onehot[train_idx], y[train_idx]
    X_cont_test, X_onehot_test, y_test = X_cont[test_idx], X_onehot[test_idx], y[test_idx]

    X_cont_train = sc.fit_transform(X_cont_train.reshape(-1,1))
    X_cont_test = sc.transform(X_cont_test.reshape(-1,1))

    X_train = np.hstack([X_cont_train, X_onehot_train])
    X_test = np.hstack([X_cont_test, X_onehot_test])
    
    print(X_train.shape, y_train.shape)

    ## Fit grid search
    best_model = gcv.fit(X_train, y_train)

    if save_trained_model:
        joblib.dump(sc, f"{result_dir}sc_trained_{ik_fold}.joblib")
        joblib.dump(best_model.best_estimator_, f"{result_dir}trained_model_{ik_fold}.joblib")

    ##---------------------------------
    ## best hyper parameters
    print('best_hyper_parameters:', best_model.best_params_)

    results = pd.DataFrame(gcv.cv_results_).sort_values(by='mean_test_score', ascending=False)
    results = results.loc[:, ~results.columns.str.endswith("_time")]
    results = results.loc[:, ~results.columns.str.startswith("split")]

    results.to_csv(f"{result_dir}hyper_params_{ik_fold}.csv", index=None)

    ## pred and probability:
    #y_pred = best_model.best_estimator_.predict(X_test)
    y_prob = best_model.best_estimator_.predict_proba(X_test)

    pred_idx = np.argmax(y_prob, axis=1)
    
    acc = np.mean(y_test == pred_idx)
    print("acc:", acc)
    
    if iik == 0:
        y_test_all = y_test
        y_prob_all = y_prob
    else:
        y_test_all = np.append(y_test_all, y_test)
        y_prob_all = np.vstack([y_prob_all,y_prob])
        
##--------------------------
pred_idxs = np.argmax(y_prob_all, axis=1)
acc_all = np.mean(y_test_all == pred_idxs)
print("acc_all:", acc_all)

# %%
##====================================================================================
df_meta = df_data

for ik_fold in np.arange(5):
    test_idx = train_valid_test_idx["test_idx"][ik_fold]
    
    if ik_fold == 0:
        test_idx_all = test_idx
    else:
        test_idx_all = np.hstack((test_idx_all,test_idx))

df_test = df_meta.loc[test_idx_all].reset_index(drop=True)

##------------------------------------------
## find topk_prediction
def find_topk_idxs(pred_scores, topk=5):
    ## increased sort
    i = np.argsort(pred_scores,axis=1) 

    ## decreased sort
    i1 = np.flip(i, axis=1)

    ## select top k
    top_idx = i1[:, :topk]
    
    return top_idx

topk_idxs = find_topk_idxs(y_prob_all, topk=5)
#print(topk_idxs.shape)

all_topk_pred_names = class_names[topk_idxs]
#print(all_topk_pred_names.shape)

for i in range(3):
    df_test[f"top{i+1}_pred_name"] = all_topk_pred_names[:,i]

df_test[class_names] = y_prob_all
##--------------------------------------------

## get the same order of slide_selected file
df_test = df_test.sort_values(by="slide_name", ignore_index=True)
df_test.to_csv(f"{result_dir}{model_name}_scores.csv", index=None)

# %%
np.savetxt(f"{result_dir}{model_name}_acc.txt", [acc_all], fmt="%s")
# %%
import numpy as np
import pandas as pd
import os,sys
import gc
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import StratifiedKFold
import joblib
sys.path.append(new_module_path)
from indirect_model_multiclass import *
from utils import *

np.random.seed(42)


run_on_computer = False

if run_on_computer:
    model_name = "LR"
    ik_fold = 0
    res_dir = "results"
    project = "NCI"
else:
    model_name = sys.argv[1]
    ik_fold = int(sys.argv[2])
    res_dir = sys.argv[3]
    project = sys.argv[4]

os.makedirs(res_dir, exist_ok=True)

path2data = "data/"
split_file = f"{path2data}{project}_train_valid_test_idx.npz"

path_to_predicted_methylation = f"{path2data}NCI_beta_pred_10class.pkl"

X = pd.read_pickle(path_to_predicted_methylation) 

y = X['class'].copy()

X.drop(columns = ['class'], inplace = True)

result_dir = f'{res_dir}/{model_name}/{ik_fold}/'
os.makedirs(result_dir, exist_ok=True)

X_train, y_train, X_test, y_test = split_data(path2data,split_file,X.reset_index(drop = True).T,y,ik_fold)


X_train.columns = X.index.values[X_train.columns]

X_test.columns = X.index.values[X_test.columns]

sc = MinMaxScaler()
X_train_columns = X_train.columns
X_train_index = X_train.index
X_test_columns = X_test.columns
X_test_index = X_test.index
X_train_t = sc.fit_transform(X_train.T)
X_test_t = sc.transform(X_test.T)
joblib.dump(sc, f'{result_dir}MinMaxScaler.joblib')

# +
X_train = pd.DataFrame(X_train_t, columns=X_train_index, index = X_train_columns)
X_test = pd.DataFrame(X_test_t, columns=X_test_index, index = X_test_columns)

#================================================================================================
feat_sel = SelectKBest(f_classif, k=1000).fit(X_train, y_train)

best_feat = feat_sel.get_feature_names_out()
np.savetxt(f"{result_dir}best_feat.txt", best_feat, fmt="%s")

X_train = X_train.loc[:,best_feat].copy()
X_test = X_test.loc[:,best_feat].copy()

model, hyper_parameters = multi_class(model_name)
# +
gcv = GridSearchCV(model, hyper_parameters, 
                   scoring='f1_micro',
                   cv=5, n_jobs=-1)

best_model = gcv.fit(X_train, y_train)
# -
joblib.dump(best_model.best_estimator_, f'{result_dir}best_model.joblib')


# +
y_pred = best_model.best_estimator_.predict(X_test)

y_prob = best_model.best_estimator_.predict_proba(X_test)
# -


df_scores = pd.DataFrame(y_prob,
                         index = X_test.index,
                         columns=best_model.best_estimator_.classes_)

df_pred = pd.DataFrame(y_pred,
                         index = X_test.index,
                         columns=['predicted'])

df_scores = df_pred.join(df_scores)

# +
label = y_test.to_frame()

label.columns = ['label']

df_scores = label.join(df_scores)
# -

df_scores.to_pickle(f"{result_dir}df_res.pkl")

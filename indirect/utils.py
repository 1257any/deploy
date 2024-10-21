# %%
import numpy as np

from sklearn.utils import shuffle


##=============================================================================
def load_data(path2data, data_file):
    Xy = np.load(f"{path2data}{data_file}", allow_pickle=True)

    X = Xy["data"]
    y = Xy["target"]

    print(f"X.shape: {X.shape}, y.shape: {y.shape}")

    return X, y

##=============================================================================
def split_data(path2data,split_file,X,y,ik_fold):

    train_test_idx = np.load(f"{split_file}", allow_pickle=True)

    train_idx = train_test_idx["train_idx"][ik_fold]
    test_idx = train_test_idx["test_idx"][ik_fold]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    if X_test.ndim == 1:
        X_test = X_test[np.newaxis,:]
        y_test = np.array([y_test])

    print(f"X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}")
    print(f"X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}")

    return X_train, y_train, X_test, y_test

# #=============================================================================

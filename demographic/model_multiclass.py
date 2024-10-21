#!/usr/bin/env python
# coding: utf-8
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier,LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
##================================================================================================
def multi_class(model_name):
    print("model_name:", model_name)

    ##------------------------------------
    if model_name == "LR": ## genetic logistic regression

        model = LogisticRegression(max_iter=10000, class_weight="balanced")

        # regularization penalty space
        penalty = ["l2"]
        #penalty = ['l2']
        #penalty = ['l1']
        #penalty = ['elasticnet']

        # solver
        #solver=['lbfgs']
        solver = ["liblinear"]
        #solver=['liblinear',"saga"]
        #solver=['saga']

        # regularization hyperparameter space
        #C = np.logspace(0.,3.,11)
        C = np.array([0.01, 0.1, 1, 10, 20, 50, 100])
        #C = np.array([1, 10])

        # l1_ratio
        #l1_ratio = np.linspace(0,1,11)

        # Create hyperparameter options
        hyper_parameters = dict(penalty=penalty,
            solver=solver,
            C=C,
            #l1_ratio=l1_ratio,
            )

    ##------------------------------------
    if model_name == "XGB":
        ## model
        model = XGBClassifier(n_estimators=100,tree_method = 'auto',use_label_encoder=False,\
                              objective='multi:softmax',eval_metric='mlogloss')

        #max_depth = np.array([4,6,8,10,12,14,16,18])
        max_depth = np.linspace(4,18,8).astype(int)

        min_child_weight = np.linspace(0,1,11)

        #l2 regularization term
        reg_lambda = np.logspace(-3,1,num=5)

        #l1 regularization term
        reg_alpha = np.logspace(-3,1,num=5) 

        ##-------------------------------------
        # Create hyperparameter options
        hyper_parameters = dict(max_depth = max_depth,min_child_weight = min_child_weight,\
                                reg_lambda = reg_lambda, reg_alpha = reg_alpha)


    ##------------------------------------
    if model_name == "RF":

        model = RandomForestClassifier(bootstrap = True, class_weight="balanced")
            
        criterion = ["gini"]
            
        # Number of trees in random forest
        #n_estimators = [int(x) for x in np.linspace(start = 10, stop = 100, num = 10)]
        n_estimators = [1000]
        #n_estimators = [10]

        # Number of features to consider at every split
        max_features = ['sqrt']

        # Maximum number of levels in tree
        max_depth = np.linspace(4,10,7).astype(int)
        #max_depth.append(None)

        # Minimum number of samples required at each leaf node
        min_samples_leaf = np.linspace(4,20,9).astype(int)

        # Minimum number of samples required to split a node
        min_samples_split = np.linspace(4,20,9).astype(int)

        # Method of selecting samples for training each tree
        #bootstrap = [True, False]
        #bootstrap = [True]

        # Create the random grid
        hyper_parameters = {
                        'criterion': criterion,
                        'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_leaf': min_samples_leaf,
                       'min_samples_split': min_samples_split,
                       #'bootstrap': bootstrap
                       }

    ##------------------------------------
    if model_name == "SVM":
        model = SVC(probability=True,class_weight="balanced", tol=0.001)

        #kernel
        #kernel = ['linear','poly','rbf','sigmoid']
        kernel = ['rbf','sigmoid']
        
        # regularization penalty space
        C = np.logspace(0,6,10)
        #C = np.logspace(0,4,10)

        # gamma
        #gamma = ['scale','auto']
        gamma = ['auto']
        
        # Create hyperparameter options
        #hyperparameters = dict(penalty=penalty,solver=solver,C=C,l1_ratio=l1_ratio)
        hyper_parameters = dict(kernel=kernel,C=C,gamma=gamma)
       
    ##------------------------------------
    if model_name == "KNN":
        model = KNeighborsClassifier()

        metric = ['minkowski','euclidean','manhattan']

        n_neighbors = np.linspace(5,30,26).astype(int)

        weights = ['uniform','distance']

        leaf_size = np.linspace(1,10,10).astype(int)
        
        # Create hyperparameter options
        hyper_parameters = dict(
                            metric = metric,
                            n_neighbors=n_neighbors,
                            weights=weights,
                            leaf_size=leaf_size,
                            )



    return model, hyper_parameters

##================================================================================================

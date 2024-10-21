# %%
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
##================================================================================================
def multi_class(model_name):
    print("model_name:", model_name)

    ##------------------------------------

    ##------------------------------------
    if model_name == "RF":

        model = RandomForestClassifier(bootstrap = True, class_weight = 'balanced')
            
        criterion = ["gini"]
            
        # Number of trees in random forest
        n_estimators = [200]

        # Number of features to consider at every split
        max_features = ['sqrt']
        # Maximum number of levels in tree
        max_depth = np.linspace(8,30,12).astype(int)

        # Minimum number of samples required at each leaf node
        min_samples_leaf = np.linspace(1,10,10).astype(int)

        # Minimum number of samples required to split a node
        min_samples_split = np.linspace(10,30,11).astype(int)


        # Create the random grid
        hyper_parameters = {
                        'criterion': criterion,
                        'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_leaf': min_samples_leaf,
                       'min_samples_split': min_samples_split,
                       }

    ##------------------------------------
    if model_name == "SVM":
        model = SVC(probability=True, class_weight = 'balanced')

        #kernel
        kernel = ['linear','poly','rbf','sigmoid']
        # regularization penalty space
        C = np.logspace(0,4,40)
        # gamma
        gamma = ['scale','auto']
        
        # Create hyperparameter options
        #hyperparameters = dict(penalty=penalty,solver=solver,C=C,l1_ratio=l1_ratio)
        hyper_parameters = dict(kernel=kernel,C=C,gamma=gamma)

    ##------------------------------------

    if model_name == "LR": ## genetic logistic regression
        
        model = LogisticRegression(max_iter=1000, class_weight="balanced")

        penalty = ['l1','l2']

        solver=["saga"]
        # regularization hyperparameter space
        C = np.logspace(0.,3.,11)
        # l1_ratio
        l1_ratio = np.linspace(0,1,11)

        # Create hyperparameter options
        hyper_parameters = dict(penalty=penalty,
            solver=solver,
            C=C
            )

        
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



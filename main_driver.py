#This file is just provided incase you want to do a quick check without running a jupyter notebook

import pandas as pd 
import numpy as np

from sklearn import preprocessing
from logistic.logistic import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import KFold



if __name__ == '__main__':

    #Load and normalize the data 
    data = pd.read_csv('data.csv')
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(data)
    scaled_data = pd.DataFrame(x_scaled)

    #Train and validation

    X = scaled_data[[0,1,2,3]].to_numpy()
    Y = scaled_data[4].to_numpy()

    kf = KFold(n_splits=5)
    iters = 0
    f1_scores = []
    auc_scores = []
    for train_index, test_index in kf.split(X):
        print("TRAIN:", len(train_index), "TEST:", len(test_index))
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        logit = LogisticRegression(4,regularization = 'L2', learningRate = 0.001,lambda_ = 0.001)
        print("split %d"%(iters+1))
        logit.SGD(X_train ,y_train.reshape(-1,1),600001 , printFreq = 50000)
        y_pred = logit.predict(X_test) 
        preds = []
        for i in y_pred:
            if i > 0.5:
                preds.append(1)
            else:
                preds.append(0) 
        iters+=1
        f1_scores.append(f1_score(preds,y_test))
        auc_scores.append(roc_auc_score(preds,y_test))
    
    print("5 fold cross valudated F1 score %f"%np.mean(f1_scores))
    print("5 fold cross valudated AUC %f"%np.mean(auc_scores))



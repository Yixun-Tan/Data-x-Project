
import pandas as pd
import numpy as np
import datetime
import os

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB # Gaussian Naive Bays

from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

import xgboost as xgb

def cv_5_default_models(X, y, test_X, ID):
    now = datetime.datetime.today().strftime('%Y-%m-%d %H_%M')
    identifier = '/Default_models'
    folder = 'Submission' + identifier + '/' + now
    os.makedirs(folder)
    
    cv = StratifiedKFold(n_splits=5)
    
    clf = {'log_reg': LogisticRegression(), 'knn': KNeighborsClassifier(n_neighbors = 3),\
           'svc': SVC(probability = True), 'random_forest': RandomForestClassifier(n_estimators=100), 'xgb': xgb.XGBClassifier(n_estimators=500)}
    aucs = {}
    accu = {}
    for name, model in clf.items():
        classifier = model
        accu[name] = []
        aucs[name] = []
        
        scores = cross_val_score(classifier, X, y, cv =5)
        accu[name] = scores
        print('5 fold CV accuracy for %s is %0.2f (+/- %0.2f)' %(name, accu[name].mean(), accu[name].std() * 2))
        
        for train, test in cv.split(X, y):
            probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
            fpr, tpr, _ = roc_curve(y[test], probas_[:, 1])
            roc_auc = auc(fpr, tpr)
            aucs[name].append(roc_auc)
        aucs[name] = np.array(aucs[name])
        print('5 fold CV ROC_AUC for %s is %0.2f (+/- %0.2f)' %(name, aucs[name].mean(), aucs[name].std() * 2))
        
        probas_ = classifier.fit(X, y).predict_proba(test_X)[:, 1]
        submit = ID
        submit['TARGET'] = probas_
        submit.to_csv(f'{folder}/{name}.csv', index = False)
        

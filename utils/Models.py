
import pandas as pd
import numpy as np
import datetime
import os
import gc

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB # Gaussian Naive Bays
from sklearn.utils import class_weight

from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold

import xgboost as xgb
from lightgbm import LGBMClassifier


def cv_5_default_models(X, y, test_X, app_test):
    
    now = datetime.datetime.today().strftime('%Y-%m-%d %H_%M')
    identifier = '/Default_models'
    folder = 'Submission' + identifier + '/' + now
    os.makedirs(folder)
    
    cv = StratifiedKFold(n_splits=5)
    
    clf = {'log_reg': LogisticRegression(), 
           #'knn': KNeighborsClassifier(n_neighbors = 3, n_jobs = -1),\
           #'svc': SVC(probability = True), 
           'random_forest': RandomForestClassifier(n_estimators = 100, random_state = 50, n_jobs = -1), 'xgb': xgb.XGBClassifier(n_estimators=500, n_jobs = -1)}
    aucs = {}
    accu = {}
    for name, model in clf.items():
        classifier = model
        accu[name] = []
        aucs[name] = []
        
        scores = cross_val_score(classifier, X, y, cv =5)
        accu[name] = scores
        print('5 fold CV accuracy for %s is %0.2f (+/- %f)' %(name, accu[name].mean(), accu[name].std() * 2))
        
        for train, test in cv.split(X, y):
            classifier.fit(X[train], y[train])
            probas_ = classifier.predict_proba(X[test])
            fpr, tpr, _ = roc_curve(y[test], probas_[:, 1])
            roc_auc = auc(fpr, tpr)
            aucs[name].append(roc_auc)
        aucs[name] = np.array(aucs[name])
        print('5 fold CV ROC_AUC for %s is %0.2f (+/- %0.2f)' %(name, aucs[name].mean(), aucs[name].std() * 2))
        
        probas_ = classifier.fit(X, y).predict_proba(test_X)[:, 1]
        
        submit = app_test[['SK_ID_CURR']]
        submit['TARGET'] = probas_
        submit.to_csv(f'{folder}/{name}.csv', index = False)

def kfold_lightgbm(df, num_folds, debug = True, name = 'Not specified', importance = True, AUC = False):
    
    now = datetime.datetime.today().strftime('%Y-%m-%d %H_%M')
    identifier = 'Lightgbm'
    folder = 'Submission' + identifier + '/' + now
    os.makedirs(folder)
    
    train_df = df[df['TARGET'] != -999]
    test_df = df[df['TARGET'] == -999]
    
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    
    del df
    gc.collect()
    
    cv = StratifiedKFold(n_splits = num_folds, shuffle = True)
    
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    
    feats = [f for f in train_df.columns if f not in ['TARGET', 'SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    
    
    for n_fold, (train_idx, valid_idx) in enumerate(cv.split(train_df[feats], train_df['TARGET'])):
        
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]
        
        
        clf = LGBMClassifier(
            n_estimators=10000,
            learning_rate=0.03,
            num_leaves=30,
            colsample_bytree=.8,
            subsample=.9,
            max_depth=7,
            reg_alpha=.1,
            reg_lambda=.1,
            min_split_gain=.01,
            min_child_weight=2,
            silent=-1,
            verbose=-1,)
        
        clf.fit(train_x, train_y, eval_set = [(train_x, train_y), (valid_x, valid_y)], eval_metric = 'auc', verbose = 200, early_stopping_rounds = 200)
        
        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration = clf.best_iteration_)[:, 1]
        # Calculate results for test set every fold and take the mean value.
        sub_preds += clf.predict_proba(test_df[feats], num_iteration = clf.best_iteration_)[:, 1] / cv.n_splits
        
        if importance:   
            fold_importance_df = pd.DataFrame()
            fold_importance_df['feature'] = feats
            fold_importance_df['importance'] = clf.feature_importances_
            fold_importance_df['fold'] = n_fold + 1
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis = 0)

        print('Fold %2d AUC: %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        
        del clf, train_x, train_y, valid_x, valid_y
        
        gc.collect()
        
    print('Full AUC score: %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    if AUC:
        Full_AUC = roc_auc_score(train_df['TARGET'], oof_preds)
    # Debug == True, dont generate submission file. Else, the opposite.
    if not debug:
        test_df['TARGET'] = sub_preds
        test_df[['SK_ID_CURR', 'TARGET']].to_csv(f'{folder}/{name}.csv', index = False)
        
    if importance and AUC: 
        display_importances(feature_importance_df)
        feature_importance_df = feature_importance_df[['feature', 'importance']].groupby('feature', as_index = False).mean().sort_values('importance', ascending = False)
        
        return feature_importance_df, Full_AUC
                                 
    if importance and not AUC:
        display_importances(feature_importance_df)
        feature_importance_df = feature_importance_df[['feature', 'importance']].groupby('feature', as_index = False).mean().sort_values('importance', ascending = False)
        
        return feature_importance_df   
    if not importance and AUC:                             
        return Full_AUC
                                 
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[['feature', 'importance']].groupby('feature').mean().sort_values(by = 'importance', ascending = False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize = (8, 10))
    sns.barplot(x = 'importance', y = 'feature', data = best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    



        

# Data manipulation
import pandas as pd
import numpy as np

# XGBoost
import xgboost as xgb

# Evaluation of the model
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

# Helper
import random
import csv
from hyperopt import STATUS_OK
from timeit import default_timer as timer

import warnings
warnings.filterwarnings('ignore')

# Import data
df = pd.read_csv('./data/final_dataset_train.csv')
y_train = df['TARGET']
x_train = df.drop(['TARGET', 'SK_ID_CURR'], axis = 1)

print('x_train.shape: {}'.format(x_train.shape))
print('y_train.shape: {}'.format(y_train.shape))
del df

# Baseline
clr_xgb = xgb.XGBClassifier(n_estimators=500, n_jobs = -1)
roc_auc_scores = cross_val_score(clr_xgb, x_train, y_train, cv=5, scoring='roc_auc')
print('Baseline Model: 5 fold CV ROC_AUC is %0.2f (+/- %0.2f)' %(roc_auc_scores.mean(), roc_auc_scores.std() * 2))

param_grid = {
        'learning_rate': list(np.logspace(np.log(0.01), np.log(0.2), base = 10, num = 1000)),
        'min_child_weight': list(np.linspace(1, 9, dtype=int)),
        'gamma': list(np.linspace(0.0, 1.0, 10)),
        'subsample': list(np.linspace(0.5, 1.0, 10)),
        'colsample_bytree': list(np.linspace(0.5, 1.0, 10)),
        'max_depth': list(np.linspace(3, 10, dtype=int)),
        }

global  ITERATION
ITERATION = 0
N_FOLDS = 5
MAX_EVALS = 100

def objective(hyperparameters):
    
    global ITERATION
    
    ITERATION += 1
        
    start = timer()
    clf = xgb.XGBClassifier(**hyperparameters, n_estimators=500, n_jobs = -1)
    # Perform n_fold cross validation
    roc_auc_scores = cross_val_score(clf, x_train, y_train, cv = N_FOLDS, scoring = 'roc_auc')
    
    run_time = timer() - start
    
    # Extract the score
    score = roc_auc_scores.mean()
    
    # Write to the csv file ('a' means append)
    of_connection = open(OUT_FILE, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([hyperparameters, ITERATION, run_time, score])
    of_connection.close()
    
    return [score , hyperparameters, ITERATION]

def random_search(param_grid, max_evals = MAX_EVALS):
    """Random search for hyperparameter optimization"""
    
    # Dataframe for results
    results = pd.DataFrame(columns = ['score', 'params', 'iteration'],
                                  index = list(range(MAX_EVALS)))
    
    # Keep searching until reach max evaluations
    for i in range(MAX_EVALS):
        
        # Choose random hyperparameters
        hyperparameters = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}

        # Evaluate randomly selected hyperparameters
        eval_results = objective(hyperparameters)
        
        results.loc[i, :] = eval_results
    
    # Sort with best score on top
    results.sort_values('score', ascending = False, inplace = True)
    results.reset_index(inplace = True)
    
    return results 

# Create a file and open a connection
OUT_FILE = 'XGB/RandomSearch_Tuning_XGB.csv'
of_connection = open(OUT_FILE, 'w')
writer = csv.writer(of_connection)

# Write column names
headers = ['hyperparameters', 'iteration', 'runtime', 'score']
writer.writerow(headers)
of_connection.close()



random_results = random_search(param_grid)

print('The best validation score was {:.5f}'.format(random_results.loc[0, 'score']))
print('\nThe best hyperparameters were:')

import pprint
pprint.pprint(random_results.loc[0, 'params'])

# Get the best parameters
random_search_params = random_results.loc[0, 'params']

# Create, train, test model
clf_tuned = xgb.XGBClassifier(**random_search_params, n_estimators=500, n_jobs = -1, random_state = 42)
# Perform n_fold cross validation
roc_auc_scores_tuned = cross_val_score(clf_tuned, x_train, y_train, cv = N_FOLDS, scoring = 'roc_auc')
print('5 fold CV ROC_AUC is %0.2f (+/- %0.2f)' %(roc_auc_scores_tuned.mean(), roc_auc_scores.std() * 2))


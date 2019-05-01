
# Data manipulation
import pandas as pd
import numpy as np

# Random Forest Model
import xgboost as xgb

# Evaluation of the model
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('Feature/final_dataset_train.csv')
y_train = df['TARGET']
x_train = df.drop(['TARGET', 'SK_ID_CURR'], axis = 1)

clr_xgb = xgb.XGBClassifier(n_estimators=500, n_jobs = -1)
roc_auc_scores = cross_val_score(clr_xgb, x_train, y_train, cv=5, scoring='roc_auc')
print('5 fold CV ROC_AUC is %0.2f (+/- %0.2f)' %(roc_auc_scores.mean(), roc_auc_scores.std() * 2))

import csv
from hyperopt import STATUS_OK
from timeit import default_timer as timer

def objective(hyperparameters):
    
    global ITERATION
    
    ITERATION += 1
        
    start = timer()
    clf = xgb.XGBClassifier(**hyperparameters, n_jobs = -1)
    # Perform n_fold cross validation
    roc_auc_scores = cross_val_score(clf, x_train, y_train, cv = N_FOLDS, scoring = 'roc_auc')
    
    run_time = timer() - start
    
    # Extract the best score
    score = roc_auc_scores.mean()
    
     # Loss must be minimized
    loss = 1 - score
    
    # Write to the csv file ('a' means append)
    of_connection = open(OUT_FILE, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss, hyperparameters, ITERATION, run_time, score])
    of_connection.close()
    
    return {'loss': loss, 'hyperparameters': hyperparameters, 'iteration': ITERATION, 'train_time': run_time, 'status': STATUS_OK}

from hyperopt import hp
from hyperopt.pyll.stochastic import sample

space = {
        'leaning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
        'min_child_weight': hp.choice('min_child_weight', np.arange(1, 10, dtype=int)),
        'gamma': hp.uniform('gamma', 0.0, 1.0),
        'subsample': hp.uniform('subsample', 0.5, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
        'max_depth': hp.choice('max_depth', np.arange(3, 10, dtype=int)),
        }

from hyperopt import tpe

# Create the algorithm
tpe_algorithm = tpe.suggest

from hyperopt import Trials

# Record results
trials = Trials()

# Create a file and open a connection
OUT_FILE = 'XGBoost/Automated_Tuning.csv'
of_connection = open(OUT_FILE, 'w')
writer = csv.writer(of_connection)

# Write column names
headers = ['loss', 'hyperparameters', 'iteration', 'runtime', 'score']
writer.writerow(headers)
of_connection.close()

from hyperopt import fmin

global  ITERATION

ITERATION = 0
# Governing choices for search
N_FOLDS = 5
MAX_EVALS = 100

best = fmin(fn = objective, space = space, algo = tpe.suggest, trials = trials,
            max_evals = MAX_EVALS)

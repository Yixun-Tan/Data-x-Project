
# Data manipulation
import pandas as pd
import numpy as np

# Modeling
import lightgbm as lgb

# Evaluation of the model
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')

# Dataset Creation
df = pd.read_csv('Ethan_google/final_dataset_train.csv')
y_train = df['TARGET']
x_train = df.drop(['TARGET', 'SK_ID_CURR'], axis = 1)
train_set = lgb.Dataset(x_train, label = y_train)

# Baseline
base_hyp = {'n_estimators': 1000, 'verbose' = -1, 'silent' = -1}
cv_results = lgb.cv(base_hyp, train_set, num_boost_round = 10000, nfold = N_FOLDS, early_stopping_rounds = 100, metrics = 'auc')
best_score = cv_results['auc-mean'][-1]
std = cv_results['auc-stdv'][-1]
print('5 fold CV ROC_AUC is %0.5f (+/- %0.2f)' %(best_score, std * 2))

import csv
from hyperopt import STATUS_OK
from timeit import default_timer as timer

import csv
from hyperopt import STATUS_OK
from timeit import default_timer as timer

def objective(hyperparameters):
    
    global ITERATION
    
    ITERATION += 1
    
    for parameter_name in ['n_estimators', 'num_leaves', 'max_depth']:
        hyperparameters[parameter_name] = int(hyperparameters[parameter_name])
        
    start = timer()
    
    # Perform n_fold cross validation
    cv_results = lgb.cv(hyperparameters, train_set, num_boost_round = 10000, nfold = N_FOLDS, early_stopping_rounds = 100, metrics = 'auc')
    
    run_time = timer() - start
    
    # Extract the best score
    best_score = cv_results['auc-mean'][-1]
    
     # Loss must be minimized
    loss = 1 - best_score
    
    # Write to the csv file ('a' means append)
    of_connection = open(OUT_FILE, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss, hyperparameters, ITERATION, run_time, best_score])
    of_connection.close()
    
    return {'loss': loss, 'hyperparameters': hyperparameters, 'iteration': ITERATION, 'train_time': run_time, 'status': STATUS_OK}


from hyperopt import hp
from hyperopt.pyll.stochastic import sample

space = {
    'n_estimators': hp.quniform('n_estimators', 1000, 20000, 1000)
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
    'num_leaves': hp.quniform('num_leaves', 20, 150, 1),
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0),
    'subsample': hp.uniform('subsample', 0.6, 1.0),
    'max_depth': hp.quniform('max_depth', 5, 15, 1),
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 0.5),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 0.5),
    'min_split_gain': hp.uniform('min_split_gain', 0.0, 0.5),
    'min_child_weight': hp.choice('min_child_weight', np.arange(1, 100, dtype = int)),
    'is_unbalance': hp.choice('is_unbalance', [True, False]),
}

from hyperopt import tpe

# Create the algorithm
tpe_algorithm = tpe.suggest

from hyperopt import Trials

# Record results
trials = Trials()

# Create a file and open a connection
import os
PATH = 'Lightgbm'
if not os.path.exists(PATH):
    os.mkdir(PATH)
OUT_FILE = PATH + '/Automated_Tuning.csv'
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

import pickle
with open('Trial_100.pkl', 'wb') as file:
    pickle.dump(trials, file)

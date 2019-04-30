
# Data manipulation
import pandas as pd
import numpy as np

# Random Forest Model
from sklearn.ensemble import RandomForestClassifier

# Evaluation of the model
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/home/Ethan_google/final_dataset_train.csv')
y_train = df['TARGET']
x_train = df.drop(['TARGET', 'SK_ID_CURR'], axis = 1)

import csv
from hyperopt import STATUS_OK
from timeit import default_timer as timer

def objective(hyperparameters):
    
    global ITERATION
    
    ITERATION += 1
        
    start = timer()
    clf = RandomForestClassifier(**hyperparameters, n_jobs = -1)
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

space = {
        'bootstrap': hp.choice('bootstrap', [True, False]),
        'max_depth': hp.choice('max_depth', np.arange(1, 100, dtype=int)),
        'max_features': hp.choice('max_features', ['auto', 'log2']),
        'min_samples_leaf': hp.choice('min_samples_leaf', np.arange(1, 12, dtype=int)),
        'min_samples_split': hp.choice('min_samples_split', np.arange(1, 12, dtype=int)),
        'n_estimators': hp.choice('n_estimators', np.arange(50, 1000, dtype=int))
        }

from hyperopt import tpe

# Create the algorithm
tpe_algorithm = tpe.suggest

from hyperopt import Trials

# Record results
trials = Trials()

# Create a file and open a connection
OUT_FILE = 'Random_Forest/Automated_Tuning_py.csv'
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

with open('Trial_RF_100.pkl', 'wb') as file:
    pickle.dump(trials, file)
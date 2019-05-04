# Data manipulation
import pandas as pd
import numpy as np

# Modeling
import lightgbm as lgb

# Evaluation of the model
from sklearn.metrics import roc_auc_score

import random
import csv
from hyperopt import STATUS_OK
from timeit import default_timer as timer

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('./data/final_dataset_train.csv')
y_train = df['TARGET']
x_train = df.drop(['TARGET', 'SK_ID_CURR'], axis = 1)
train_set = lgb.Dataset(x_train, label = y_train)

# Create a file and open a connection
import os
PATH = 'Lightgbm'
if not os.path.exists(PATH):
    os.mkdir(PATH)
OUT_FILE = PATH + '/RandomSearch_Tuning.csv'
of_connection = open(OUT_FILE, 'w')
writer = csv.writer(of_connection)

# Write column names
headers = ['hyperparameters', 'iteration', 'runtime', 'score']
writer.writerow(headers)
of_connection.close()

# Hyperparameter grid
param_grid = {
    'n_estimators': list(np.linspace(1000, 20000, 1000, dtype=int)),
    'learning_rate': list(np.logspace(np.log(0.01), np.log(0.2), base = 10, num = 1000)),
    'num_leaves': range(20, 150),
    'reg_alpha': list(np.linspace(0, 0.5)),
    'reg_lambda': list(np.linspace(0, 0.5)),
    'colsample_bytree': list(np.linspace(0.6, 1, 10)),
    'subsample': list(np.linspace(0.6, 1, 100)),
    'is_unbalance': [True, False],
    'boosting_type': ['gbdt', 'goss', 'dart'],
    'max_depth': range(5, 15),
    'min_split_gain': list(np.linspace(0.0, 0.5))
}

# Baseline
random.seed(42)

# Randomly sample from dictionary
random_params = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
# Deal with subsample ratio
random_params['subsample'] = 1.0 if random_params['boosting_type'] == 'goss' else random_params['subsample']
print(random_params)

# Tuning
global  ITERATION

ITERATION = 0
MAX_EVALS = 100
N_FOLDS = 5

def objective(hyperparameters, iteration):
    """Objective function for grid and random search. Returns
       the cross validation score from a set of hyperparameters."""
    
    global ITERATION
    ITERATION += 1
    
    # Number of estimators will be found using early stopping
    #if 'n_estimators' in hyperparameters.keys():
    #    del hyperparameters['n_estimators']
        
    start = timer()
     # Perform n_folds cross validation
    cv_results = lgb.cv(hyperparameters, train_set, num_boost_round = 10000, nfold = N_FOLDS, 
                        early_stopping_rounds = 100, metrics = 'auc', seed = 42)
    run_time = timer() - start
    
    # results to retun
    best_score  = cv_results['auc-mean'][-1]
    estimators = len(cv_results['auc-mean'])
    hyperparameters['n_estimators'] = estimators 
    
    # Write to the csv file ('a' means append)
    of_connection = open(OUT_FILE, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([hyperparameters, ITERATION, run_time, best_score])
    of_connection.close()
    
    return [best_score , hyperparameters, iteration]

def random_search(param_grid, max_evals = MAX_EVALS):
    """Random search for hyperparameter optimization"""
    
    # Dataframe for results
    results = pd.DataFrame(columns = ['score', 'params', 'iteration'],
                                  index = list(range(MAX_EVALS)))
    
    # Keep searching until reach max evaluations
    for i in range(MAX_EVALS):
        
        # Choose random hyperparameters
        hyperparameters = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
        hyperparameters['subsample'] = 1.0 if hyperparameters['boosting_type'] == 'goss' else hyperparameters['subsample']

        # Evaluate randomly selected hyperparameters
        eval_results = objective(hyperparameters, i)
        
        results.loc[i, :] = eval_results
    
    # Sort with best score on top
    results.sort_values('score', ascending = False, inplace = True)
    results.reset_index(inplace = True)
    
    return results 

random_results = random_search(param_grid)

print('The best validation score was {:.5f}'.format(random_results.loc[0, 'score']))
print('\nThe best hyperparameters were:')

import pprint
pprint.pprint(random_results.loc[0, 'params'])

# Get the best parameters
random_search_params = random_results.loc[0, 'params']

# Create, train, test model
model = lgb.LGBMClassifier(**random_search_params, random_state = 42)
model.fit(train_features, train_labels)

preds = model.predict_proba(test_features)[:, 1]

print('The best model from random search scores {:.5f} ROC AUC on the test set.'.format(roc_auc_score(test_labels, preds)))
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

def missing_values_table(df):
    
    #Create a table containing missing value information
    mis_val = df.isnull().sum()
    mis_val_percent = mis_val * 100 / len(df)
    mis_val_table = pd.concat((mis_val, mis_val_percent), axis = 1)
    #print(mis_val_table)
    
    mis_val_table = mis_val_table[mis_val_table.iloc[:,1] != 0]
    mis_val_table = mis_val_table.rename(columns = {0: 'Missing Values', 1: '% of Total Values'})
    mis_val_table = mis_val_table.sort_values('% of Total Values', ascending = False).round(1)
    
    print('There are total', df.shape[1], 'columns.')
    print(mis_val_table.shape[0], 'of them have missing values.')
    return mis_val_table

def Pre_train_processing(train, test):
    imputer = SimpleImputer(strategy = 'median')
    scaler = MinMaxScaler(feature_range = (0,1))
    # Imputation
    train = imputer.fit_transform(train)
    test = imputer.transform(test)
    # Normalization
    train = scaler.fit_transform(train)
    test = scaler. transform(test)
    print('Training data shape: ', train.shape)
    print('Testing data shape: ', test.shape)
    return train, test
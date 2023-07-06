from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import argparse
from rdkit.Chem import AllChem
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import os
from tqdm import tqdm
import sys
import numpy
from matplotlib import pylab
from sklearn.model_selection import KFold
from statistics import mean
import yaml
from sklearn import metrics
def parse_args():
    """Parses arguments from cmd"""
    parser = argparse.ArgumentParser(description="Preprocess: encode property change and build vocabulary")

    # parser.add_argument("--settings", "-s", help=("Input setting name"), type=str,required=True)
    parser.add_argument("--file_name", "-f", help=("Input file name"), type=str,required=True)
    return parser.parse_args()
def toarray(fp):
    array = np.zeros((0, ), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, array)
    return array

def calc_fp(df):
    
    mol = [Chem.MolFromSmiles(smi) for smi in df['smiles']]
    fp = [AllChem.GetHashedMorganFingerprint(m,2,2048) for m in mol]
    fp = [toarray(f) for f in fp]
    return fp



def set_seed(seed = 42):
    """
    Enables reproducibility.
    
    """
    
    random.seed(seed)
    np.random.seed(seed)

    os.environ['PYTHONHASHSEED']=str(seed)

def read(file_name):
    print('Start Reading File')
    dataset_name = file_name + '.csv'
    path = "../data/"
    df = pd.read_csv(path + dataset_name)

    print('Finish reading File')
    return df



def divide_1_9th(trainset):
    length = len(trainset)
    index = round(length/9)
    val = trainset.iloc[:index, :]
    train = trainset.iloc[index:, :]
    return val, train



def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)    

def rf(train,test,validation):
    X_train = calc_fp(train)
    X_test = calc_fp(test)
    X_val = calc_fp(validation)
    y_train = train['prop']
    y_test = test['prop']
    y_val = validation['prop']
    regressor = RandomForestRegressor(random_state=0)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    y_pred_valid = regressor.predict(X_val)
    t_pred_train = regressor.predict(X_train)
    
    test_mae = metrics.mean_absolute_error(y_test, y_pred)
    test_r2 = metrics.r2_score(y_test, y_pred)
    test_rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

    train_mae = metrics.mean_absolute_error(y_train, t_pred_train)
    train_r2 = metrics.r2_score(y_train, t_pred_train)
    train_rmse = np.sqrt(metrics.mean_squared_error(y_train, t_pred_train))

    val_mae = metrics.mean_absolute_error(y_val, y_pred_valid)
    val_r2 = metrics.r2_score(y_val, y_pred_valid)
    val_rmse = np.sqrt(metrics.mean_squared_error(y_val, y_pred_valid))
    print('Mean Absolute Error:', test_mae)
    print('r squre:', test_r2)
    print('Root Mean Squared Error:', test_rmse)

    return test_mae,test_r2,test_rmse, train_mae, train_r2, train_rmse, val_mae, val_r2, val_rmse

def pool_rmse(ls):
    ls_rmse = ls
    sd = np.std(ls_rmse)
    ls_rmse = [x**2 for x in ls_rmse]
    rmse = sum(ls_rmse)/len(ls_rmse)
    rmse = math.sqrt(rmse)
    return rmse








if __name__ == "__main__":

    args = parse_args()
    file_name = args.file_name
    outpath = '../results/random_forest/'
    figpath = outpath + file_name + '/'
    tables_path = figpath + 'tables/'
    plots_path = figpath + 'plots/'
    make_dir(outpath)
    make_dir(figpath)  
    make_dir(tables_path)
    make_dir(plots_path)
    
    # settings= yaml.safe_load(open('../settings/' + args.settings, "r"))


    set_seed(seed = 42)

    file_name = file_name
    df = read(file_name)
    df.rename(columns={ df.columns[1]: "prop" }, inplace = True)
    df = df.sample(frac=1).reset_index(drop=True)
    result = pd.DataFrame(columns=['fold','r2','rmse'])
    result['fold'] = list(range(10)) + ['mean']
    k_folds = 10
    skf = KFold(n_splits = k_folds, shuffle = False)
    r2 = []
    rmse = []
    for k, (k_train, k_val) in enumerate(skf.split(df.iloc[:,:])):
        # Generate k-fold
        k_train = df.iloc[k_train, :]
        k_val = df.iloc[k_val,:]
        k_validation, k_train = divide_1_9th(k_train)
        k_train.reset_index(drop = True, inplace = True)
        k_val.reset_index(drop = True, inplace = True)
        k_validation.reset_index(drop = True, inplace = True)
        k_train.insert(0, 'ID', k_train.index)
        k_val.insert(0, 'ID', k_val.index)
        k_validation.insert(0, 'ID', k_validation.index)

        test_mae,test_r2,test_rmse, train_mae, train_r2, train_rmse, val_mae, val_r2, val_rmses = rf(k_train,k_val,k_validation)
        r2.append(test_r2)
        rmse.append(test_rmse)
    print(r2)
    r2.append(mean(r2))
    rmse.append(pool_rmse(rmse))
    result['r2'] = r2
    result['rmse'] = rmse


    result.to_csv(f'{tables_path}{file_name}.csv')





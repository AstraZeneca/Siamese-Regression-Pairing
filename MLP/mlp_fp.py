import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit.Chem import AllChem
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import numpy
import sys
from scipy.stats import norm
from matplotlib import pylab
from matplotlib.pyplot import figure
from sklearn.model_selection import KFold
import module.SimpleNeuron_copy as cc
from statistics import mean
import os
import argparse
import yaml
def parse_args():
    """Parses arguments from cmd"""
    parser = argparse.ArgumentParser(description="Preprocess: encode property change and build vocabulary")

    parser.add_argument("--settings", "-s", help=("Input setting name"), type=str,required=True)
    parser.add_argument("--file_name", "-f", help=("Input file name"), type=str,required=True)
    return parser.parse_args()
def set_seed(seed = 42):
    """
    Enables reproducibility.
    
    """
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED']=str(seed)

def read(file_name):
    print('Start Reading File')
    dataset_name = file_name + '.csv'
    path = "../data/"
    df = pd.read_csv(path + dataset_name)

    print('Finish reading File')
    return df

def calculate_delta_fp(df):
    df['mol1'] = df['comp1'].map(lambda x: Chem.MolFromSmiles(x))
    df['mol2'] = df['comp2'].map(lambda x: Chem.MolFromSmiles(x))
    df['fp1'] = [AllChem.GetHashedMorganFingerprint(mol,2,2048) for mol in df['mol1']]
    df['fp2'] = [AllChem.GetHashedMorganFingerprint(mol,2,2048) for mol in df['mol2']]
    df['fp'] = df['fp1'] - df['fp2']
    return df
def calc_pool_rmse(rmse_ls):
    rmse = [x**2 for x in rmse_ls]
    rmse = sum(rmse)/len(rmse)
    rmse = math.sqrt(rmse)
    return rmse

def divide_1_9th(trainset):
    length = len(trainset)
    index = round(length/9)
    val = trainset.iloc[:index, :]
    train = trainset.iloc[index:, :]
    return val, train



def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)  


if __name__ == "__main__":
    args = parse_args()
    file_name = args.file_name
    outpath = '../results/MLP_fp/'
    figpath = outpath + file_name + '/'
    tables_path = figpath + 'tables/'
    plots_path = figpath + 'plots/'
    make_dir(outpath)
    make_dir(figpath)  
    make_dir(tables_path)
    make_dir(plots_path)
    settings= yaml.safe_load(open('../settings/' + args.settings, "r"))

    rmse_0 = []
    rmse_3 = []
    rmse_35 =[]
    rmse_4 = []
    rmse_45 = []
    rmse_5 = []
    r2_0 = []
    r2_3 = []
    r2_35 =[]
    r2_4 = []
    r2_45 = []
    r2_5 = []
    set_seed(seed = 42)

    file_name = file_name
    df = read(file_name)
    df = df.sample(frac=1).reset_index(drop=True)
    k_folds = 10
    skf = KFold(n_splits = k_folds, shuffle = False)
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
        print('Fold {}/{} ...'.format(k+1, k_folds))
        rmse_ls, r2_ls = cc.train_classic(
            k, 
            plots_path, 
            k_train, 
            k_val, 
            settings['lr'],
            settings['patience'],
            settings['factor'],
            settings['min_lr'],
            settings['eps'],
            settings['epochs'],
            val_set = k_validation,
            file_name = file_name)
        rmse_0.append(rmse_ls[0])
        rmse_3.append(rmse_ls[1])
        rmse_35.append(rmse_ls[2])
        rmse_4.append(rmse_ls[3])
        rmse_45.append(rmse_ls[4])
        rmse_5.append(rmse_ls[5])
        r2_0.append(r2_ls[0])
        r2_3.append(r2_ls[1])
        r2_35.append(r2_ls[2])
        r2_4.append(r2_ls[3])
        r2_45.append(r2_ls[4])
        r2_5.append(r2_ls[5])


    rmse_0 = calc_pool_rmse(rmse_0)
    rmse_3 = calc_pool_rmse(rmse_3)
    rmse_35 = calc_pool_rmse(rmse_35)
    rmse_4 = calc_pool_rmse(rmse_4)
    rmse_45 = calc_pool_rmse(rmse_45)
    rmse_5 = calc_pool_rmse(rmse_5)

    r2_0 = mean(r2_0)
    r2_3 = mean(r2_3)
    r2_35 = mean(r2_35)
    r2_4 = mean(r2_4)
    r2_45 = mean(r2_45)
    r2_5 = mean(r2_5)
    print('classic RMSE:\n0: {}\n0.3: {}\n0.35: {}\n0.4: {}\n0.45: {}\n0.5: {}'.format(rmse_0, rmse_3, rmse_35, rmse_4, rmse_45,rmse_5))
    df = pd.DataFrame()
    df['cutoff'] = [0,0.3,0.35,0.4,0.45,0.5]
    df['rmse'] = [rmse_0, rmse_3, rmse_35, rmse_4, rmse_45, rmse_5]
    df['r2'] = [r2_0, r2_3, r2_35, r2_4, r2_45, r2_5]
    df.to_csv(tables_path+'avg_res.csv')





    


import torch
import argparse
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
from tqdm import tqdm
import sys
import numpy
import sys
from matplotlib import pylab
from sklearn.model_selection import KFold
from statistics import mean
import module.paired_train as pt
import module.predict_fold as p
import yaml
# def parse_args():
#     """Parses arguments from cmd"""
#     parser = argparse.ArgumentParser(description="Preprocess: encode property change and build vocabulary")

#     parser.add_argument("--file_name", "-f", help=("Input file name"), type=str,required=True)
#     parser.add_argument("--strat", "-s", help=("Input pairing strategy: 0 for all pairs, 1 for top 1 and so on"), type=int,required=True)
#     parser.add_argument("--lr", "-lr", help = ("Input learning rate"), type = float)
#     parser.add_argument("--patience", "-p", help = ("Input patience of adaptive learning"), type = float)
#     parser.add_argument("--factor", "-fc", help = ("Input factor of adaptive learning"), type = float)
#     parser.add_argument("--min_lr", "-ml", help = ("Input minimum learning rate of adaptive learning"), type = float)
#     parser.add_argument("--eps", "-ml", help = ("Input eps of adaptive learning"), type = float)
#     parser.add_argument("--epochs", "-ml", help = ("Input number of epochs"), type = int)

#     return parser.parse_args()
def parse_args():
    """Parses arguments from cmd"""
    parser = argparse.ArgumentParser(description="Preprocess: encode property change and build vocabulary")

    parser.add_argument("--settings", "-s", help=("Input setting name"), type=str,required=True)
    parser.add_argument("--file_name", "-f", help=("Input file name"), type=str,required=True)
    parser.add_argument("--strat", "-st", help=("Input pairing strategy: 0 for all pairs, 1 for top 1 and so on"), type=int,required=True)
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
    strategy = args.strat
    outpath = '../results/MLP_delta/top ' + str(strategy) + '/'
    figpath = outpath + file_name + '/'
    tables_path = figpath + 'tables/'
    plots_path = figpath + 'plots/'
    make_dir(outpath)
    make_dir(figpath)  
    make_dir(tables_path)
    make_dir(plots_path)
    
    settings= yaml.safe_load(open('../settings/' + args.settings, "r"))


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
        pair_ob = pt.paired_trainer(
            plots_path,
            k_train,
            k_val,
            strategy,
            settings['lr'],
            settings['patience'],
            settings['factor'],
            settings['min_lr'],
            settings['eps'],
            settings['epochs'],
            val_set = k_validation,
            file_name = file_name)
        
        pair_train = pair_ob.train(k, 128, plots_path)
        pair_result = pair_ob.predict_test()
        pair_result.to_csv(tables_path + str(k) + '.csv')


    cutoff = [0,0.3,0.35,0.4,0.45,0.5]
    shots = list(range(1,21))
    p.predict(plots_path, tables_path, cutoff, shots, args.file_name)





    

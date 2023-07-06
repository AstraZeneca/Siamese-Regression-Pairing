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
sys.path.append('../MLP/module/')
from pair_generation_top import Generator 
import numpy
from matplotlib import pylab
from sklearn.model_selection import KFold
from statistics import mean
import yaml
from sklearn import metrics
import predict_fold as p
def toarray(fp):
    array = np.zeros((0, ), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, array)
    return array
def to_fp_ECFP(smi):
    mol = Chem.MolFromSmiles(smi)
    return AllChem.GetHashedMorganFingerprint(mol,2,2048)
def progress_bar(iterable, total, **kwargs):
    return tqdm.tqdm(iterable=iterable, total=total, ascii=True, **kwargs)

def sim(smi1, smi2):
    return DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi1), 2), AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi2), 2))
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

def get_delta_fp(df, fp_pair1, fp_pair2):
    ID1 = df['ID1']
    ID2 = df['ID2']
    fp1 = [fp_pair1[x] for x in ID1]
    fp2 = [fp_pair2[x] for x in ID2]
    delta_fp = np.subtract(np.array(fp1), np.array(fp2))
    X = np.array([toarray(fp) for fp in delta_fp])
    return X


def test_train_pair(self):
    df = pd.DataFrame(columns = ['ID1', 'ID2','comp1','comp2', 'sim', 'comprop2','prop'])
    for i in range(len(self.test_set)):
        temp = pd.DataFrame(columns = ['ID1','comp1','comp2', 'sim', 'comprop2','prop'])
        temp['ID1'] = [i] *len(self.train_set)
        temp['ID2'] = self.train_set.iloc[:, 0]
        temp['comp1'] = [self.test_set.iloc[i, -2]] * len(self.train_set)
        temp['comp2'] = self.train_set.iloc[:, -2]
        temp['sim'] = [DataStructs.TanimotoSimilarity(self.test_fp[i] , x) for x in self.train_fp]
        temp['comprop2'] = self.train_set.iloc[:, -1]
        temp['prop'] = [self.test_set.iloc[i,-1] - x for x in temp['comprop2']]
        df = pd.concat([df, temp], ignore_index=True)

    return df

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)    

class rf(object):
    def __init__(self,train,test,validation, file_name, top):
        self.train_set = train
        self.test_set = test
        self.val_set = validation
        self.train_fp = [to_fp_ECFP(smi) for smi in self.train_set['smiles']]
        self.test_fp = [to_fp_ECFP(smi) for smi in self.test_set['smiles']]
        # self.val_fp = [to_fp_ECFP(smi) for smi in self.val_set['smiles']]
        self.train_pair = Generator(file_name, df = train, top = top).maxoutput
        self.val_pair = Generator(file_name, df = validation, top = top).maxoutput

        
    def train(self):
        y_train = self.train_set['prop']
        y_train_delta = self.train_pair['prop']
        y_test = self.test_set['prop']
        # y_val = self.val_set['prop']
        # y_val_delta = self.val_pair['prop']
        print('calculating delta fingerprints')
        X_train = get_delta_fp(self.train_pair, self.train_fp, self.train_fp)
        print('making train_test pairs')
        test_train_pair = self.test_train_pair()
        print('calculating delta fingerprints for train_test pairs')
        X_test = get_delta_fp(test_train_pair, self.test_fp, self.train_fp)
        regressor = RandomForestRegressor(random_state=0)
        print('training rf')
        regressor.fit(X_train, y_train_delta)
        print('prediction')
        test_train_pair['pred_prop']  = regressor.predict(X_test)
        print('finish prediction')
        return test_train_pair
        


    def test_train_pair(self):
        df = pd.DataFrame(columns = ['ID1', 'ID2','comp1','comp2', 'sim', 'comprop2','prop'])
        for i in range(len(self.test_set)):
            temp = pd.DataFrame(columns = ['ID1','comp1','comp2', 'sim', 'comprop2','prop'])
            temp['ID1'] = [i] *len(self.train_set)
            temp['ID2'] = self.train_set.iloc[:, 0]
            temp['comp1'] = [self.test_set.iloc[i, -2]] * len(self.train_set)
            temp['comp2'] = self.train_set.iloc[:, -2]
            temp['sim'] = [DataStructs.TanimotoSimilarity(self.test_fp[i] , x) for x in self.train_fp]
            temp['comprop2'] = self.train_set.iloc[:, -1]
            temp['prop'] = [self.test_set.iloc[i,-1] - x for x in temp['comprop2']]
            df = pd.concat([df, temp], ignore_index=True)
        return df





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
    outpath = '../results/random_forest_delta/'
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
        print(f'Processing {k} fold')
        k_train = df.iloc[k_train, :]
        k_val = df.iloc[k_val,:]
        k_validation, k_train = divide_1_9th(k_train)
        k_train.reset_index(drop = True, inplace = True)
        k_val.reset_index(drop = True, inplace = True)
        k_validation.reset_index(drop = True, inplace = True)
        k_train.insert(0, 'ID', k_train.index)
        k_val.insert(0, 'ID', k_val.index)
        k_validation.insert(0, 'ID', k_validation.index)
        test_train_pair = rf(k_train,k_val,k_validation, file_name, 1).train()
        test_train_pair.to_csv(tables_path + str(k) + '.csv')

    cutoff = [0,0.3,0.35,0.4,0.45,0.5]
    shots = list(range(1,21))
    p.predict(plots_path, tables_path, cutoff, shots, args.file_name)





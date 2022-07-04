import pandas as pd
import splitter as sp
import pair_generation_top as pg
import os
from sklearn.model_selection import KFold
import numpy as np

def make_dir(file_name):
    ouput_path =  file_name + '/'
    try: 
        os.mkdir(ouput_path) 
    except OSError as error: 
        print(error)  
    return ouput_path

k_folds = 10
skf = KFold(n_splits = k_folds, shuffle = False)
def divide_1_9th(trainset):
    length = len(trainset)
    index = round(length/9)
    val = trainset.iloc[:index, :]
    train = trainset.iloc[index:, :]
    return val, train

def make_input_k_fold(file_name, sample):
    df1 = pd.read_csv('data/' + file_name + '.csv')
    df1 = df1.sample(frac=1).reset_index(drop=True)
    for k, (k_train, k_val) in enumerate(skf.split(df1)):
        train_set = df1.iloc[k_train, :]
        test_set = df1.iloc[k_val,:]
        val_set, train_set = divide_1_9th(train_set)
        train_set.reset_index(drop = True, inplace = True)
        test_set.reset_index(drop = True, inplace = True)
        val_set.reset_index(drop = True, inplace = True)
        train_set['SET'] = 'train'
        test_set['SET'] = 'test'
        val_set['SET'] = 'valid'
        df_ls = [train_set] * sample + [test_set, val_set]
        df =  pd.concat(df_ls)
        df.rename(columns={ df.columns[-2]: "prop" }, inplace = True)
        df['smiles'] = ['<' + file_name + '>|' + smi for smi in  df['smiles']]
        df['data'] = file_name
        yield df

lipo = list(make_input_k_fold('lipo', 1))
freesolv = list(make_input_k_fold('freesolv', 1))
delaney = list(make_input_k_fold('delaney', 1))

for i in range(10):
    ouput_path = make_dir('lipo')
    lipo[i].to_csv(ouput_path + str(i) + '.csv')
    ouput_path = make_dir('freesolv')
    freesolv[i].to_csv(ouput_path + str(i) + '.csv')
    ouput_path = make_dir('delaney')
    delaney[i].to_csv(ouput_path + str(i) + '.csv')

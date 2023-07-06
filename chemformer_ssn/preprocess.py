import pandas as pd
import splitter as sp
import modules.pair_generation_top as pg
import os
from sklearn.model_selection import KFold
import numpy as np
def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)  


def make_dir_file(file_name, random  = False):
    if random:
        ouput_path =  'data/random/' + file_name + '/'
    else: 
        ouput_path =  'data/similarity/' + file_name + '/'

    try: 
        make_dir(ouput_path) 
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
def make_input_k_fold(file_name, sample, random = False):
    df1 = pd.read_csv('../data/' + file_name + '.csv')
    df1 = df1.sample(frac=1).reset_index(drop=True)
    for k, (k_train, k_val) in enumerate(skf.split(df1)):
        train_set = df1.iloc[k_train, :]
        test_set = df1.iloc[k_val,:]
        val_set, train_set = divide_1_9th(train_set)
        train_set.reset_index(drop = True, inplace = True)
        test_set.reset_index(drop = True, inplace = True)
        val_set.reset_index(drop = True, inplace = True)
        if random == False:
            train_set = pg.Generator(file_name, train_set, 1).maxoutput
            test_set = pg.Generator(file_name, test_set, 1).maxoutput
            val_set = pg.Generator(file_name, val_set, 1).maxoutput
        else:
            train_set = pg.Generator(file_name, train_set, 1, random = True).maxoutput
            test_set = pg.Generator(file_name, test_set, 1, random = True).maxoutput
            val_set = pg.Generator(file_name, val_set, 1, random = True).maxoutput
        train_set['SET'] = 'train'
        test_set['SET'] = 'test'
        val_set['SET'] = 'valid'
        df_ls = [train_set] * sample + [test_set, val_set]
        df =  pd.concat(df_ls)
        df['comp1'] = ['<' + file_name + '>|' + smi for smi in df['comp1']]
        df['comp2'] = ['<' + file_name + '>|' + smi for smi in df['comp2']]
        df['data'] = file_name
        yield df

def preprocess(file_name, top, random):
    ls = list(make_input_k_fold(file_name, top, random))
    for i in range(10):
        ouput_path = make_dir_file(file_name, random)
        ls[i].to_csv(ouput_path + str(i) + '.csv')    





if __name__ ==  '__main__':
    preprocess('lipo',1, False)
    preprocess('freesolv',1, False)
    preprocess('delaney',1, False)
    preprocess('lipo',1, True)
    preprocess('freesolv',1, True)
    preprocess('delaney',1, True)

# lipo = list(make_input_k_fold('lipo', 1))
# freesolv = list(make_input_k_fold('freesolv', 1))
# ESOL = list(make_input_k_fold('delaney', 1))

# for i in range(10):
#     ouput_path = make_dir('lipo')
#     lipo[i].to_csv(ouput_path + str(i) + '.csv')
#     ouput_path = make_dir('freesolv')
#     freesolv[i].to_csv(ouput_path + str(i) + '.csv')
#     ouput_path = make_dir('delaney')
#     ESOL[i].to_csv(ouput_path + str(i) + '.csv')

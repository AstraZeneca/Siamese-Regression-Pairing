import pandas as pd
from rdkit import Chem

from rdkit import DataStructs
import matplotlib.pyplot as plt
import molvs as mv
from rdkit.Chem import RDKFingerprint
import numpy as np
from tqdm import tqdm
import os
from multiprocessing import Pool, cpu_count
import math
import time
import itertools
import datetime
from rdkit.Chem import AllChem


def to_fp_ECFP(smi):
    if smi:
        try:
            mol = Chem.MolFromSmiles(smi)
        except Exception:
            print(smi)
        if mol is None:
            return None
        return AllChem.GetHashedMorganFingerprint(mol,2,2048)
def tanimoto_similarity_pool(args):
    return tanimoto_similarity(*args)
def tanimoto_similarity(id1, id2):
    smi1, smi2 = id_simi_dict[id1], id_simi_dict[id2]
    fp1, fp2 = None, None
    if smi1 and type(smi1)==str and len(smi1)>0:
        fp1 = to_fp_ECFP(smi1)
    if smi2 and type(smi2)==str and len(smi2)>0:
        fp2 = to_fp_ECFP(smi2)
    if fp1 is not None and fp2 is not None:
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    else:
        return None

class Generator(object):
    def __init__(self, file_name, df = np.nan, top = 100, random = False, sample_size = 50):
        self.file_name = file_name
        self.top = top
        self.random = random
        self.sample_size = sample_size
        if isinstance(df, float) == True:
            self.df = self.read()
        else:
            self.df = df
        self.originaldf = self.process_df()
        self.simdf = self.similarity()
        self.maxoutput = self.maxmatrix()

    
    def read(self):
        print('Start Reading File')
        dataset_name = self.file_name + '.csv'
        #path = "datasets/standardised_datasets/"
        df = pd.read_csv(dataset_name)
        print('Finish reading File')
        return df

    def process_df(self):
        df1 = self.df.copy()
        df1['mol'] = df1['smiles'].map(lambda x: Chem.MolFromSmiles(x))

        df1.insert(0, 'ID', range(len(df1)))
        fp = [to_fp_ECFP(smi) for smi in df1['smiles']]
        self.fp = fp
        print(len(self.fp))
        #df1.to_csv('empty.csv')
        return df1

    
    def rowcalc(self, fps_column, fps_row):
        return DataStructs.TanimotoSimilarity(fps_row, fps_column)
    
    def calc(self, i):
        fp = self.fp
        orgdf = self.originaldf
        fps_row =fp[i] 
        fps_column = fp[i+1:]
        result = [self.rowcalc(fps_row, x) for x in fps_column]
        if result != []:
            result = [np.nan]*(i+1) + result
            max_index = sorted(range(len(result)), key=lambda a: result[a])[-1*self.top:]
            max_value = [result[b] for b in max_index]
            
            ID2 = list(orgdf.iloc[max_index,0])
        else:
            max_value = np.nan
            max_index = np.nan
        ID1 = [orgdf.iloc[i,0]] * self.top
        comprop1 = [orgdf.iloc[i,-2]] * self.top
        comprop2 = list(orgdf.iloc[max_index,-2])
        comp1 = [orgdf.iloc[i,1]]* self.top
        comp2 = list(orgdf.iloc[max_index,1])
        return (ID1, ID2, max_value, comp1, comp2, comprop1, comprop2)
    
    def random_calc(self, i):
        sample_size = self.sample_size
        fp = self.fp
        orgdf = self.originaldf
        fps_row =fp[i] 
        fps_column = fp[i+1:]
        result = [self.rowcalc(fps_row, x) for x in fps_column]
        if result != []:
            result = [np.nan]*(i+1) + result
            max_index = pd.Series(result).sample(n=sample_size, replace=False).index
            max_value = [result[b] for b in max_index]
            
            ID2 = list(orgdf.iloc[max_index,0])
        else:
            max_value = np.nan
            max_index = np.nan
        ID1 = [orgdf.iloc[i,0]] * sample_size
        comprop1 = [orgdf.iloc[i,-2]] * sample_size
        comprop2 = list(orgdf.iloc[max_index,-2])
        comp1 = [orgdf.iloc[i,1]]* sample_size
        comp2 = list(orgdf.iloc[max_index,1])
        return (ID1, ID2, max_value, comp1, comp2, comprop1, comprop2)








    def similarity(self):
        print('Start Generating Similarity Matrix Values')
        start_time = time.time()
        orgdf = self.originaldf
        i = range(len(orgdf['ID'])-1)
        print(datetime.datetime.now(), 'Start computing similarity----------------')
        with Pool(cpu_count()-1) as p:
            if self.random:
                results = p.map(self.random_calc, i)
            else:
                results = p.map(self.calc, i)
        print(datetime.datetime.now(), 'End computing similarity')
        df = pd.DataFrame()
        ID1 = []
        ID2 = []
        sim = []
        c1 = []
        c2 = []
        p1 = []
        p2 = []
        for item in results:

            ID1 += item[0]
            ID2 += item[1]
            sim += item[2]
            c1 += item[3]
            c2 += item[4]
            p1 += item[5]
            p2 += item[6]

        df['ID1'] = ID1
        df['ID2'] = ID2
        df['Similarity'] = sim
        df['comp1'] = c1
        df['comp2'] = c2
        df['comprop1'] = p1
        df['comprop2'] = p2

        print('Finish Generating Similarity Matrix Values')
        print("--- %s seconds ---" % (time.time() - start_time))
        #print(df)

        return df






    def maxmatrix(self):
        simdf = self.simdf.copy()
        # originaldf = self.originaldf
        # prop = list(originaldf.columns.values)[2:-1]
        print('Start Generating Pair Matrix')
        simdf['prop'] = simdf['comprop1'] - simdf['comprop2']

        # simdf.drop(columns = ['comprop1', 'comprop2'], inplace = True)
        simdf.dropna(inplace = True)
        print('Finish Generating Pair Matrix')
        return simdf


















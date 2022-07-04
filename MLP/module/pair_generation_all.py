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
    def __init__(self, file_name, df = np.nan):
        self.file_name = file_name
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
        df = pd.read_csv(dataset_name)
        print('Finish reading File')
        return df

    def process_df(self):
        df1 = self.df.copy()
        df1['mol'] = df1['smiles'].map(lambda x: Chem.MolFromSmiles(x))

        df1.columns.values[0] = 'ID'
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
        fps_row =fp[i] # change to morgan fingerprint, move out of the loop
        fps_column = fp[i+1:]
        length = fps_column
        result = [self.rowcalc(fps_row, x) for x in fps_column]
        ID1 = [orgdf.iloc[i,0]] * len(length)
        ID2 = list(orgdf.iloc[i+1:,0])
        comprop1 = [orgdf.iloc[i,-2]] * len(length)
        comprop2 = list(orgdf.iloc[i+1:,-2])
        comp1 = [orgdf.iloc[i,1]]* len(length)
        comp2 = list(orgdf.iloc[i+1:,1])
        return (ID1, ID2, result, comp1, comp2, comprop1, comprop2)
    

    def similarity(self):
        print('Start Generating Similarity Matrix Values')
        start_time = time.time()
        orgdf = self.originaldf
        i = range(len(orgdf['ID'])-1)
        print(datetime.datetime.now(), 'Start computing similarity----------------')
        with Pool(cpu_count()-1) as p:
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
        df['comp1'] = c1
        df['comp2'] = c2
        df['sim'] = sim
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
        simdf['lipo'] = simdf['comprop1'] - simdf['comprop2']

        simdf.drop(columns = ['comprop1', 'comprop2'], inplace = True)
        simdf.dropna(inplace = True)

        print('Finish Generating Pair Matrix')

        return simdf

















import torch
from rdkit.Chem import AllChem
import pandas as pd
from rdkit import Chem
import numpy as np
from rdkit import DataStructs
import torch.nn as nn
import torch

def to_fp_ECFP(smi):
    mol = Chem.MolFromSmiles(smi)
    return AllChem.GetHashedMorganFingerprint(mol,2,2048)


def toarray(fp):
    array = np.zeros((0, ), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, array)
    return array

class Dataset(object):
    """The training table dataset.
    """
    def __init__(self, df, fp_pair1, fp_pair2):
        self.df = df
        ID1 = self.df['ID1']
        ID2 = self.df['ID2']
        fp1 = [fp_pair1[x] for x in ID1]
        fp2 = [fp_pair2[x] for x in ID2]
        X1 = np.array([toarray(fp) for fp in fp1])
        X2 = np.array([toarray(fp) for fp in fp2])
        Y = np.array(df.iloc[:,-1])
               
        self.x1_data = X1
        self.x2_data = X2
        self.y_data = Y 
        self.len = len(self.x1_data) # Size of data

    def __getitem__(self, index):
        return self.x1_data[index], self.x2_data[index], self.y_data[index]
        
    def __len__(self):
        return self.len

    @classmethod
    def collate_fn(cls, data_all):
        
        X1, X2, Y = zip(*data_all)
        X1 = torch.from_numpy(np.asarray(X1).astype(np.float32))
        X2 = torch.from_numpy(np.asarray(X2).astype(np.float32))
        Y = np.asarray(Y)
        Y = torch.tensor(Y.astype(np.float32).reshape(-1, 1)) 



        return X1, X2, Y

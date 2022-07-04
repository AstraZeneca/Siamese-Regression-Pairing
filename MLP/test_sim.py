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


comp1 = 'CC(C)(C)Cl'
comp2 = 'CC(C)(C)O'

fp1 = to_fp_ECFP(comp1)
fp2 = to_fp_ECFP(comp2)
print(DataStructs.TanimotoSimilarity(fp1, fp2))
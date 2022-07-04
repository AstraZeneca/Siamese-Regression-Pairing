import torch

from torch.utils.data import DataLoader
from rdkit import DataStructs
from rdkit import Chem
from multiprocessing import Pool, cpu_count
import pandas as pd
from rdkit.Chem import AllChem
from sklearn.metrics import r2_score
import math








class predict_process():
    def __init__(self, loss_df_name):

        self.file_name = loss_df_name
        self.df = self.load_loss_df()

    def load_loss_df(self):
        df = pd.read_csv(self.file_name)
        df['pred'] = df['comprop2'] + df['pred_prop']
        df['comprop1'] = df['comprop2'] + df['prop']
        return df

    def process_loss(self, cutoff, shots, small =False):
        df = self.df.copy()
        df1 = df.drop_duplicates(subset='comp1')
        ID = list(range(len(df1)))
        max_ID = max(ID)
        self.max_ID = max_ID
        tot_ID = list(range(0, int(max_ID) + 1))
        miss = 0
        mean = []
        real = []
        sd_prop = []
        for a in range(0, max_ID+1, 1):
            slc0 = df[df['ID1'] == a]
            slc0.reset_index(drop=True, inplace=True)
            if cutoff > 0: 
                slc= self.cutoff_process(slc0, cutoff, small = small)
                if slc.empty:
                    miss += 1
                    if shots > 0:
                        slc = self.shot_process(slc0, shots)
            else:
                if shots > 0:
                    slc = self.shot_process(slc0, shots)
                else:
                    slc = slc0

            mean0 = slc['pred'].mean()
            real0 = slc['comprop1'].mean()
            sd_prop0 = slc['pred'].std()
            mean.append(mean0)
            real.append(real0)
        df1 = pd.DataFrame(columns = ['mean', 'real', 'miss'])
        df1['mean'] = mean
        df1['real'] = real
        df1['miss'] = miss
        df1.dropna(inplace = True)
        df1['error'] = df1['mean'] - df1['real']
        error_squre = df1['error'] **2
        tot = error_squre.sum()/len(error_squre)
        RMSE = math.sqrt(tot)
        r2 = r2_score(df1['real'], df1['mean'])

        return RMSE, miss, r2
                
            


    def cutoff_process(self, slc, cutoff, small = False):
        if small:
            slc = slc[slc['sim'] < cutoff]
        else:
            slc = slc[slc['sim'] >= cutoff]

        return slc
    def shot_process(self, slc, shots):
        slc = slc.sort_values('sim',ascending=False)
        slc = slc.iloc[:shots, :]
        return slc
    def plot(self, cutoff):
        df = self.df.copy()
        df = df[df['sim'] >= cutoff]
        sim = df['sim']
        err = df['error']
        density_scatter( np.array(sim), np.array(err),  s = 1, bins = [30,30] )
        plt.xlabel('Tanimoto Similarity')
        plt.ylabel('Prediction Error')
        plt.savefig(str(cutoff) + 'error_vs_sim_density.png')
        plt.clf()         

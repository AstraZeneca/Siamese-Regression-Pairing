import torch

from torch.utils.data import DataLoader
from rdkit import DataStructs
from rdkit import Chem
from multiprocessing import Pool, cpu_count
import pandas as pd
from rdkit.Chem import AllChem
from sklearn.metrics import r2_score
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
from statistics import mean
def density_scatter( x , y, ax = None, sort = True, bins = 20, **kwargs )   :
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None :
        fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y, c=z, **kwargs )
    norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    #cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
    
    #cbar.ax.set_ylabel('Density')

    return ax







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
    def plot(self, cutoff, k):
        df = self.df.copy()
        df = df[df['sim'] >= cutoff]
        sim = df['sim']
        err = df['pred_prop'] - df['prop']
        fig, ax = plt.subplots()
        increment = 0.2
        intervals = np.arange(0, 1.2, 0.2)
        averged_error_per_bin = []
        for element in intervals:
            average = cal_sd(df, element, element + increment)
            averged_error_per_bin.append(average)
        nega = [-1 * x for x in averged_error_per_bin]
        ax.plot(intervals, averged_error_per_bin, 'r--', label = r'2$\sigma$')
        ax.plot(intervals, nega, 'r--')
        ax.legend()
        # for element in intervals:
        #     ax.vlines(element, ymin=err.min(), ymax=err.max(),
        #             linestyles="dashed",
        #             colors="grey",
        #             linewidth=1.5)
        ax.set_xlabel('Tanimoto Similarity')
        ax.set_xlim(0,1.05)
        divider = make_axes_locatable(ax)
        ax.set_ylabel('Prediction Error')
        ax_histx = divider.append_axes("top", '20%', pad="3%", sharex=ax)
        ax_histy = divider.append_axes("right", '20%', pad="3%", sharey=ax)
        ax_histx.tick_params(axis="x", labelbottom=False, direction='in')
        ax_histy.tick_params(axis="y", labelleft=False, direction='in')
        ax_histx.hist(df['sim'], bins=50,density=True, orientation='vertical') #,zorder=0)
        ax_histy.hist(err, bins=50,density=True, orientation='horizontal') #,zorder=0)
        density_scatter( np.array(sim), np.array(err),  s = 1, bins = [30,30], ax = ax )
        ax.xaxis.set_ticks(np.arange(0, 1.1, 0.2))
        ax.set_xlim(-0.1, 1.1)        
        ax.yaxis.set_ticks(np.arange(-5, 11, 5))
        ax.set_ylim(-7, 12)
        plt.ylabel('Prediction Error')
        plt.tight_layout()
        plt.savefig('../results/MLP_delta/error_vs_sim_density.png')
        plt.clf()     

def cal_sd(df1, sim1, sim2):
    df = df1[(df1['sim'] >= sim1) & (df1['sim'] <= sim2)]
    sd = []
    df2 = df.drop_duplicates(subset='comp1')
    ID = list(range(len(df2)))
    max_ID = max(ID)
    for a in tqdm(range(0, max_ID+1, 1)):
        slc0 = df[df['ID1'] == a]
        slc0.reset_index(drop=True, inplace=True)
        sd_prop0 = slc0['pred'].std()
        sd.append(sd_prop0*2)
    sd = [x for x in sd if not math.isnan(x)]
    av = mean(sd)
    return av

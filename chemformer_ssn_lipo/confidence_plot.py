
import os
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

from statistics import mean
import math


import statistics
from sklearn.metrics import r2_score
import argparse

class predict_process():
    def __init__(self, loss_df_name):
        self.file_name = loss_df_name
        self.df = self.load_loss_df()

    def load_loss_df(self):
        df = pd.read_csv(self.file_name)
        return df

    def process_loss(self, cutoff, shots, small =False):
        df = self.df.copy()
        print(df)
        df['test_prop'] = pd.to_numeric(df['comprop1'].astype(str).str.strip(), errors='coerce')
        test_ID = pd.unique(df['ID1'])
        num_test = len(test_ID)
        self.max_ID = num_test
        miss = 0
        mean = []
        real = []
        sd_prop = []
        sim = []
        for a in test_ID:
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
            real0 = slc['test_prop'].mean()
            sd_prop0 = slc['pred'].std()
            mean.append(mean0)
            real.append(real0)
            sd_prop.append(sd_prop0)
            sim.append(slc['sim'].mean())
        df1 = pd.DataFrame(columns = ['mean', 'real', 'miss', 'sd'])
        df1['mean'] = mean
        df1['real'] = real
        df1['miss'] = miss
        df1['sd'] = sd_prop
        df1['sim'] = sim
        df1.dropna(inplace = True)
        df1['error'] = df1['mean'] - df1['real']
        df1['error'] =  df1['error'].abs()

        return df1
                
            


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








def rmse(ls):
    error_squre = [x ** 2 for x in ls]
    tot = mean(error_squre)
    RMSE = math.sqrt(tot)
    return RMSE


def confidence_curve(ax, dataframe, task, mini, maxi, step):
    """
    Plots the confidence curve.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe of the results.
    task : str
        The task to consider
    mini: float
        min y aixs
    maxi: float
        max y axis
    step: float
        interval y axis


    Returns
    -------
    None
    """

    std_error_zip = list(zip(dataframe['sd'].values,
                             dataframe['error'].values,
                             dataframe['sim'].values))
    # Sort the standard deviation values in increasing order
    std_error_zip.sort(key=lambda x: x[0])

    avg_err = []
    avg_sim = []
    while std_error_zip:
        # Computer the mean on the observations

        avg_err.append(rmse([err[1] for err in std_error_zip]))
        avg_sim.append(rmse([err[2] for err in std_error_zip]))
        # Remove the observation with the highest standard deviation
        std_error_zip = std_error_zip[: -1]

    x_axis = [100 - i/len(avg_err)*100 for i in range(len(avg_err), 0, -1)]
    lns1 = ax.plot(x_axis, avg_err, label = 'RMSE', linewidth=2.5)
    

    ax2=ax.twinx()

    lns2 = ax2.plot(x_axis, avg_sim,color = "#d62728", label = 'Mean Similarity', linewidth=2.5)
    ax2.set_ylabel("Mean Similarity", fontsize = 13)
    ax2.yaxis.set_ticks(np.arange(0.3, 0.81, 0.1))
    x_ax = [i for i in range(0, 110, 10)]
    ax.set_xticks(x_ax)
    #ax.yaxis.set_ticks(np.arange(mini,maxi, step))
    lns = lns1+lns2
    labs = [l.get_label() for l in lns]
    if task == 'lipo':
        ax.legend(lns, labs, loc='lower left', fontsize=13)

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_xticklabels(x_ax[::-1])

    #ax.set_title(f"Confidence curve\nusing the Siamese-Chemformer model on the {task} data")
    if task == 'delaney':
        ax.set_xlabel("% of compounds ranked by uncertainty", fontsize = 13)
    ax.set_ylabel("RMSE", fontsize = 13)

    


def make(k, task, cutoff, shots, drp):
    path = f'../results/chemformer_snn/dropout{drp}/{task}/tables/'
    process_predict = predict_process(path + str(k) + '.csv')
    df = process_predict.process_loss(cutoff = cutoff, shots = shots, small =False)
    return df
    #scatter_std_dev_vs_error(df, task, path, save_fig=True)
    



data_ls = ['lipo', 'freesolv', 'delaney']
shot_ls = [8,4,10]      # lowest rmse shot
mini = [0.4,0.2,0.5]
maxi = [0.9,1.2,0.81]
step = [0.1,0.2,0.1]
fig,ax = plt.subplots(3, 1, sharex=True, figsize=(6,8), dpi = 300)
for k in range(3):
    data = data_ls[k]
    shot = shot_ls[k]
    mini0 = mini[k]
    maxi0 = maxi[k]
    step0 = step[k]
    path =  '../results/chemformer_snn/dropout0.0/tables' + data + '/'
    ls = []
    for i in range(10):
        temp = make(i, data, 0, shot, 0.0)
        ls.append(temp)
    df = pd.concat(ls)
    
    confidence_curve(ax[k],df,data , mini0, maxi0, step0)
plt.tight_layout()
plt.savefig('../results/chemformer_snn/dropout0.0/confidence_curve.png')




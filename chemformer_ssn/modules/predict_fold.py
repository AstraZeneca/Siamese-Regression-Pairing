import  modules.predict as p
import pandas as pd
import numpy as np
from tqdm import tqdm
import math
import matplotlib.pyplot as plt

def predict(plots_path, tables_path, cutoff_ls, shot_ls, file_name):
    RMSE = []
    miss = []
    RMSE_shot = []
    RMSE_mix = []
    r2_shot = []
    r2_cutoff = []
    fh = []
    cut = []
    shot = []
    max_ID = []
    df = pd.DataFrame()
    for i in tqdm(range(10)):
        predict_ob = p.predict_process(tables_path + str(i) + '.csv')
        for j in cutoff_ls:
            RMSE0, miss0, r20 = predict_ob.process_loss(cutoff = j, shots = 0, small =False)
            max_ID0 = predict_ob.max_ID
            max_ID.append(max_ID0)
            cut.append(j)
            fh.append(i)
            RMSE.append(RMSE0)
            miss.append(miss0)
            RMSE_shot.append('-')
            r2_cutoff.append(r20)
            r2_shot.append('-')
            shot.append(0)
        for k in shot_ls:
            RMSE1, miss1, r21= predict_ob.process_loss(cutoff = 0, shots = k, small =False)
            max_ID0 = predict_ob.max_ID
            max_ID.append(max_ID0)
            cut.append(0)
            fh.append(i)
            RMSE.append('-')
            miss.append(miss1)
            RMSE_shot.append(RMSE1)
            r2_cutoff.append('-')
            r2_shot.append(r21)
            shot.append(k)


    df['file'] = fh
    df['cutoff']= cut
    df['shot'] = shot
    df['RMSE_cutoff'] = RMSE
    df['RMSE_shot'] = RMSE_shot
    df['r2_cutoff'] = r2_cutoff
    df['r2_shot'] = r2_shot
    df['miss'] = miss
    df['num_test'] = max_ID
    df.to_csv(tables_path + 'result.csv')


    RMSE = []
    R2 = []
    SD = []
    MISS = []
    for i in cutoff_ls:
        slc = df[(df['cutoff'] == i) & (df['shot'] == 0)]
        ls_rmse = list(slc['RMSE_cutoff'])
        sd = np.std(ls_rmse)
        ls_rmse = [x**2 for x in ls_rmse]
        rmse = sum(ls_rmse)/len(ls_rmse)
        rmse = math.sqrt(rmse)
        ls_miss = list(slc['miss'])
        percentage = sum(ls_miss)/slc['num_test'].sum()
        r2 = slc['r2_cutoff'].mean()
        RMSE.append(rmse)
        R2.append(r2)
        SD.append(sd)
        MISS.append(percentage)
        print("cutoff {} pooled RMSE: {}  r2 = {}  SD = {}    Miss: {}".format(i,rmse, r2, sd, percentage))
    shot_rmse = []
    for b in shot_ls:
        slc = df[df['shot'] == b]
        ls_rmse = list(slc['RMSE_shot'])
        sd = np.std(ls_rmse)
        ls_rmse = [x**2 for x in ls_rmse]
        rmse = sum(ls_rmse)/len(ls_rmse)
        rmse = math.sqrt(rmse)
        shot_rmse.append(rmse)
        r2 = slc['r2_shot'].mean()
        RMSE.append(rmse)
        R2.append(r2)
        SD.append(sd)
        MISS.append(0)
        cutoff_ls.append(0)
        print("shots {} pooled RMSE: {}  r2 = {}  SD = {}".format(b,rmse, r2, sd))
    avg_df = pd.DataFrame()
    avg_df['cutoff'] = cutoff_ls
    avg_df['shots'] = [0]*6 + list(shot_ls)
    avg_df['rmse'] = RMSE
    avg_df['r2'] = R2
    avg_df['sd'] = SD
    avg_df['miss'] = MISS
    avg_df.to_csv(tables_path + 'avg_res.csv')
    
    plt.plot(shot_ls, shot_rmse, 'o-', label = file_name)
    plt.xticks(np.arange(0,21, 5))
    plt.ylabel('RMSE')
    plt.xlabel('Number of Reference Compound')
    plt.legend(loc = 'upper right')
    plt.tight_layout()
    plt.savefig(plots_path + 'few_shot_plot_relative.svg')
    plt.clf()


    

    







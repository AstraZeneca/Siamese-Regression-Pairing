import pandas as pd
import numpy as np
import argparse
import matplotlib.pylab as plt
from rdkit import Chem
import rdkit.Chem.Descriptors
import os

def pref_split(smiles_string):
    """
    splits a string on '|' into a list with two elements, starting from the right
    the prefix token must be in '<pref_token>'
    eg. '<pXC50><OPRM1>|N1[C@H](C(NCCCC[C@@H...=CC=C3'
    or  '<OPRM1>|N1[C@H](C(NCCCC[C@@H...=CC=C3'
    """
    part = smiles_string.rsplit('|',1) 
    return part[1]   

def rmse(ls):
    error_squre = [x ** 2 for x in ls]
    tot = mean(error_squre)
    RMSE = math.sqrt(tot)
    return RMSE


def parse_args():
    """Parses arguments from cmd"""
    parser = argparse.ArgumentParser(description="Preprocess: encode property change and build vocabulary")

    # parser.add_argument("--settings", "-s", help=("Input setting name"), type=str,required=True)
    parser.add_argument("--top", "-t", help=("Input Top n"), type=int,required=True)
    return parser.parse_args()

def make_df(df, top_n):
    result = df.drop_duplicates(subset='comp1')
    ID_ls = df['ID1'].unique()
    full = []
    top = []
    std_full = []
    std_top = []
    sd_sim_full_ls = []
    sd_sim_top_ls = []
    for i in ID_ls:
        slc = df[df['ID1'] == i]
        all_mean = slc['sim'].mean()
        sd_full = slc['pred'].std()
        sd_sim_full = slc['sim'].std()
        slc = slc.sort_values(by=['sim'], ascending = False)
        top_mean = slc['sim'].iloc[:top_n].mean()
        top_std = slc['pred'].iloc[:top_n].std()
        sd_sim_top = slc['sim'].iloc[:top_n].std()
        full.append(all_mean)
        top.append(top_mean)
        std_full.append(sd_full)
        std_top.append(top_std)
        sd_sim_full_ls.append(sd_sim_full)
        sd_sim_top_ls.append(sd_sim_top)
    result[f'top{top_n}_mean'] = top
    result['mean_sim'] = full
    result[f'top{top_n}_std'] = std_top
    result['std_sim'] = std_full
    result['sd_sim_full'] = sd_sim_full_ls
    result['sd_sim_top'] = sd_sim_top_ls
    return result

def get_x_y(df, top):
    y = df[f'top{top}_mean']
    x = df[f'top{top}_std']
    return x, y





def read(data_path, fold, top):
    df = pd.read_csv(f'{data_path}{fold}.csv')
    df['comp1'] = [pref_split(i) for i in df['comp1']]
    df.drop(columns = ['test_fp', 'train_fp'], inplace=True)
    df = make_df(df, top)
    return df



if __name__ == "__main__":

    args = parse_args()
    file_name = ['lipo', 'freesolv', 'delaney']
    top = args.top
    
    fig, ax = plt.subplots(3,1, figsize=(6,8))
    
    num = 0
    for i in file_name:
        data_path = f'../results/chemformer_snn/dropout0.0/{i}/tables/'
        df = read(data_path, 0, top)
        x,y = get_x_y(df, top)
        ax[num].scatter(x, y)

        
        ax[num].set_ylabel('Average Similarity')
        if i == 'lipo':
            ax[num].set_yticks(np.arange(0, 1.05, 0.3))
        elif i == 'freesolv':
            ax[num].set_ylim([0.0, 0.7])
            ax[num].set_yticks(np.arange(0, 0.7, 0.2))
        else:
            ax[num].set_yticks(np.arange(0, 0.9, 0.2))
            ax[num].set_xlabel('Uncertainty')

        num += 1
    plt.tight_layout()
    plt.savefig(f'uncertainty_sim.png')
    plt.clf()





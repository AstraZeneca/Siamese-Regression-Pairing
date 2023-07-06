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




def parse_args():
    """Parses arguments from cmd"""
    parser = argparse.ArgumentParser(description="Preprocess: encode property change and build vocabulary")

    # parser.add_argument("--settings", "-s", help=("Input setting name"), type=str,required=True)
    parser.add_argument("--file_name", "-f", help=("Input file name"), type=str,required=True)
    return parser.parse_args()

def make_df(df, top_n):
    result = df.drop_duplicates(subset='comp1')
    ID_ls = df['ID1'].unique()
    full = []
    top = []
    for i in ID_ls:
        slc = df[df['ID1'] == i]
        all_mean = slc['sim'].mean()
        slc = slc.sort_values(by=['sim'], ascending = False)
        top_mean = slc['sim'].iloc[:top_n].mean()
        full.append(all_mean)
        top.append(top_mean)
    result[f'top{top_n}'] = top
    result['mean sim'] = full
    result['MW'] = [Chem.Descriptors.ExactMolWt(Chem.MolFromSmiles(smi)) for smi in result['comp1']]
    result['TPSA'] = [Chem.Descriptors.TPSA(Chem.MolFromSmiles(smi)) for smi in result['comp1']]
    return result



def make_plot(sim, prop, plot_path, prop_name):
    plt.scatter(sim, prop)
    plt.xlabel('Similarity', fontsize=15)
    plt.ylabel(prop_name, fontsize=15)
    plt.tight_layout()
    plt.savefig(f'{plot_path}{prop_name}_sim.png')
    plt.clf()



def read(data_path, fold):
    df = pd.read_csv(f'{data_path}{fold}.csv')
    df['comp1'] = [pref_split(i) for i in df['comp1']]
    df.drop(columns = ['test_fp', 'train_fp'], inplace=True)
    df = make_df(df, 10)
    return df

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)   



if __name__ == "__main__":

    args = parse_args()
    file_name = args.file_name
    outpath_mean = f'../results/chemformer_snn/dropout0.0/{file_name}/plots/mean/'
    outpath_top = f'../results/chemformer_snn/dropout0.0/{file_name}/plots/top/'
    make_dir(outpath_mean)
    make_dir(outpath_top)
    data_path = f'../results/chemformer_snn/dropout0.0/{file_name}/tables/'
    df = read(data_path, 0)
    if file_name == 'lipo':
        property_name = 'LogD'
    elif file_name == 'freesolv':
        property_name = 'Free energy'
    else:
        property_name = 'Log Solubility'
    make_plot(df['mean sim'], df['comprop1'], outpath_mean, property_name)
    make_plot(df['mean sim'], df['MW'], outpath_mean, 'MW')
    make_plot(df['mean sim'], df['TPSA'], outpath_mean, 'TPSA')
    make_plot(df['top10'], df['comprop1'], outpath_top, property_name)
    make_plot(df['top10'], df['MW'], outpath_top, 'MW')
    make_plot(df['top10'], df['TPSA'], outpath_top, 'TPSA')
    


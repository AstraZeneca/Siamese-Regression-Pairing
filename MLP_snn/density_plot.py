
from rdkit.Chem import AllChem
import pandas as pd
import sys
from module.pair_generation_top import Generator
from  module.pair_generation_all import Generator as Generator_all
from  module.paired_dataset import Dataset
import math
from  statistics import mean
from rdkit import DataStructs
from rdkit import Chem
import torch.nn as nn
from module.pair_generation_top import Generator
import tqdm
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.preprocessing import MinMaxScaler
from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn
import numpy
from  module.pair_generation_all import Generator as Generator_all


plt.rcParams.update({'font.size': 15})
def toarray(fp):
    array = np.zeros((0, ), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, array)
    return array
def to_fp_ECFP(smi):
    mol = Chem.MolFromSmiles(smi)
    return AllChem.GetHashedMorganFingerprint(mol,2,2048)
def progress_bar(iterable, total, **kwargs):
    return tqdm.tqdm(iterable=iterable, total=total, ascii=True, **kwargs)

def sim(smi1, smi2):
    return DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi1), 2), AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi2), 2))

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
    ax.xaxis.set_ticks(np.arange(0, 1.2, 0.2))
    ax.set_xlim(0,1.05)
    ax.scatter( x, y, c=z, **kwargs )
    
    norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    #cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
    
    #cbar.ax.set_ylabel('Density')

    return ax


def cal_sd_delta(df1, sim1, sim2):
    df = df1[(df1['Similarity'] >= sim1) & (df1['Similarity'] <= sim2)]

    sd = df['prop'].std() * 2

    return sd
def cal_sd_pred(df1, sim1, sim2, file_name):
    df = df1[(df1['sim'] >= sim1) & (df1['sim'] <= sim2)]
    df2 = df.drop_duplicates(subset='comp1')
    if df.empty:
        return 0
    else:
        ID = list(df2['ID1'])
        sd = []
        for a in ID:
            slc0 = df[df['ID1'] == a]
            if slc0.empty:
                print(f'no ID  = {a} exists')
            slc0.reset_index(drop=True, inplace=True)
            sd_prop0 = slc0['pred_prop'].std()
            sd.append(sd_prop0*2)
        sd = [x for x in sd if not math.isnan(x)]
        if len(sd) == 0:
            slc0 = df[df['ID1'] == a]
            return slc0['pred_prop']
        else:
            av = mean(sd)
            return av




def plot(k, file_name, top):

    path = f'../results/MLP_snn/top {top}/{file_name}/tables'
    df = pd.read_csv(f'{path}/{k}.csv')
    df. tail(-1)

    train_set = df.drop_duplicates(subset='ID2').reset_index()

    train_set = train_set[['index','comp2', 'comprop2']]
    
    train_set.rename(columns={'comp2': 'smiles'}, inplace= True)
    if top ==1:
        train_pair = Generator(file_name, df = train_set, top = top).maxoutput
    else:
        train_pair = Generator_all(file_name, df = train_set).maxoutput
        train_pair = train_pair.rename({'sim': 'Similarity', 'lipo': 'prop'}, axis=1)

    train_pair.dropna(inplace = True)

    fig, axs = plt.subplots(2, figsize=(8,10))

    ### delta vs similarity
    density_scatter(train_pair['Similarity'],train_pair.iloc[:,-1], s = 3, bins = [30,30], ax = axs[0])
    increment = 0.2
    intervals = numpy.arange(0, 1.2, 0.2)
    averged_error_per_bin = []
    for element in intervals:
        average = cal_sd_delta(train_pair, element, element + increment)
        averged_error_per_bin.append(average)
    nega = [-1 * x for x in averged_error_per_bin]
    axs[0].plot(intervals, averged_error_per_bin, 'r--', label = r'2$\sigma$')
    axs[0].plot(intervals, nega, 'r--')
    # setting for freesolv

    if (file_name == 'lipo' and top == 1):
        axs[0].set_ylabel(r'Exp. $\Delta$Log D', fontsize = 15)
        # axs[0].set_ylim([-5,5])
        # axs[1].set_ylim([-15,16])

        
    elif (file_name == 'freesolv' and top == 1):
        axs[0].set_ylabel(r'Exp. $\Delta$Free energy', fontsize = 15)
        # axs[0].set_ylim([-10,11])
        # axs[1].set_ylim([-20,21])
    elif (file_name == 'delaney'  and top == 1):
        axs[0].set_ylabel(r'Exp. $\Delta$Log solubility ', fontsize = 15)
        # axs[0].set_ylim([-7,7.5])
        # axs[1].set_ylim([-10,11])
    divider = make_axes_locatable(axs[0])
    ax_histx = divider.append_axes("top", '20%', pad="3%", sharex=axs[0])
    ax_histy = divider.append_axes("right", '20%', pad="3%", sharey=axs[0])
    ax_histx.tick_params(axis="x", labelbottom=False, direction='in')
    ax_histy.tick_params(axis="y", labelleft=False, direction='in')
    ax_histx.hist(train_pair['Similarity'], bins=50,density=True, orientation='vertical') #,zorder=0)
    ax_histy.hist(train_pair.iloc[:,-1], bins=50,density=True, orientation='horizontal') #,zorder=0)

    ax_histy.set_xticks(np.arange(0, 0.31, 0.3))



    axs[0].set_xlim(-0.1, 1.1)
    axs[0].xaxis.set_ticks(np.arange(0, 1.1, 0.2), labels = [])




    ### pred error vs sim

    sim = df['sim']
    err = df['pred_prop'] - df['prop']
    increment = 0.2
    intervals = np.arange(0, 1.2, 0.2)
    averged_error_per_bin = []
    for element in intervals:
        average = cal_sd_pred(df, element, element + increment, file_name)
        averged_error_per_bin.append(average)
    nega = [-1 * x for x in averged_error_per_bin]
    axs[1].plot(intervals, averged_error_per_bin, 'r--', label = r'2$\sigma$')
    axs[1].plot(intervals, nega, 'r--')
    axs[1].legend()
    # for element in intervals:
    #     ax.vlines(element, ymin=err.min(), ymax=err.max(),
    #             linestyles="dashed",
    #             colors="grey",
    #             linewidth=1.5)
    axs[1].set_xlabel('Tanimoto Similarity')
    axs[1].set_xlim(-0.1, 1.1)
    divider = make_axes_locatable(axs[1])
    axs[1].set_ylabel('Prediction Error')
    ax_histx = divider.append_axes("top", '20%', pad="3%", sharex=axs[1])
    ax_histy = divider.append_axes("right", '20%', pad="3%", sharey=axs[1])
    ax_histx.tick_params(axis="x", labelbottom=False, direction='in')
    ax_histy.tick_params(axis="y", labelleft=False, direction='in')
    ax_histx.hist(df['sim'], bins=50,density=True, orientation='vertical') #,zorder=0)
    ax_histy.hist(err, bins=50,density=True, orientation='horizontal') #,zorder=0)
    density_scatter( np.array(sim), np.array(err),  s = 1, bins = [30,30], ax = axs[1] )
    #  settings for freesolv
    axs[1].xaxis.set_ticks(np.arange(0, 1.1, 0.2))
    axs[1].set_xlim(-0.1, 1.1)        
 
    plt.tight_layout()
    plt.savefig(f'{file_name}_top{top}_density_{k}.png')
    plt.clf()     


plot(0, 'freesolv', 1)

plot(0, 'lipo', 1)

plot(0, 'delaney', 1)

plot(0, 'freesolv', 0)

plot(0, 'lipo', 0)

plot(0, 'delaney', 0)

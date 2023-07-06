import matplotlib.pyplot as plt
import numpy as np
import matplotlib.style
import matplotlib as mpl
import pandas as pd
# fontsize
plt.rcParams.update({'font.size': 18})
cutoff = [0,0.3,0.35,0.4,0.45,0.5]

def get_res(dataset, model):
    if model == 'Chemformer':
        path = f'../results/Chemformer/single_task/{dataset}/tables/'
        df = pd.read_csv(path + 'avg_res.csv')
    if model == 'chemformer_snn':
        path = f'../results/chemformer_snn/dropout0.0/{dataset}/tables/'
        df = pd.read_csv(path + 'avg_res.csv')
        df = df[df['shots'] == 0]
    

    if model == 'MLP_delta':
        path = f'../results/MLP_delta/top 1/{dataset}/tables/'
        df = pd.read_csv(path + 'avg_res.csv')
        df = df[df['shots'] == 0]

    if model == 'MLP_fp':
        path = f'../results/MLP_fp/{dataset}/tables/'
        df = pd.read_csv(path + 'avg_res.csv')

    if model == 'MLP_snn':
        path = f'../results/MLP_snn/top 1/{dataset}/tables/'
        df = pd.read_csv(path + 'avg_res.csv')       

    return list(df['rmse']), list(df['r2'])
model_list = ['MLP_fp', 'MLP_delta', 'MLP_snn','Chemformer', 'chemformer_snn' ]
data_list = ['lipo','freesolv', 'delaney']

res_df = pd.DataFrame()
dic = {}


for i in data_list:
    rmse = []
    r2 = []
    for k in model_list:
        rmse0, r20 = get_res(i, k)
        rmse.append(rmse0)
        r2.append(r20)
    dic[i] = [rmse,r2]


       


def make_res(ax, ymin, ymax, step, metrics, dataset):
    if metrics =='r2' :
        ax.set_ylabel(r'$r^{2}$', fontsize = 16)
        res_list = dic[dataset][1]
        
    else:
        ax.set_ylabel('RMSE', fontsize = 16)
        res_list = dic[dataset][0]

    
    ax.plot(cutoff, res_list[0], 'o-', color = 'tab:blue', label = 'MLP-FP')
    ax.plot(cutoff, res_list[1], 's--', color = 'tab:blue', label = r'MLP-$\Delta$FP')
    ax.plot(cutoff, res_list[2], 's--', color = 'tab:blue', label = 'MLP-snn')
    ax.plot(cutoff, res_list[3], '*-', color = "#d62728", label = 'Chemformer')
    ax.plot(cutoff, res_list[4], 'h--',  color = "#d62728", label = 'Chemformer-SNN')
    ax.yaxis.set_ticks(np.arange(ymin, ymax, step))

    if dataset == 'delaney':
        ax.set_xlabel('Similarity cutoff', fontsize = 16)
    return ax



fig,ax= plt.subplots(3, 2, sharex=True, dpi = 300, figsize = (15,10))
ax[0,0] = make_res(ax[0,0], 0.5, 0.91, 0.1, metrics = 'rmse', dataset = 'lipo')
ax[1,0] = make_res(ax[1,0], 0.6, 2.6, 0.4, metrics = 'rmse', dataset = 'freesolv')
ax[2,0] = make_res(ax[2,0], 0.5, 1.3, 0.2, metrics = 'rmse', dataset = 'delaney')

ax[0,1] = make_res(ax[0,1], 0.4, 0.9, 0.1, metrics = 'r2', dataset = 'lipo')
ax[1,1] = make_res(ax[1,1], 0.6, 1, 0.1, metrics = 'r2', dataset = 'freesolv')
ax[2,1] = make_res(ax[2,1], 0.7, 1, 0.1, metrics = 'r2', dataset = 'delaney')



ax[1,1].legend(fontsize=14)
plt.tight_layout()
plt.savefig('full_comparison.svg')
plt.clf()
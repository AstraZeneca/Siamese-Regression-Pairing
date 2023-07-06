import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
x = ['0','0.05', '0.1', '0.17']
def get_res(dataset):
    rmse = []
    r2 = []
    for i in x:
        path = f'../results/chemformer_snn/dropout{str(float(i))}/{dataset}/tables/'
        df = pd.read_csv(path + 'avg_res.csv')
        df = df[(df['cutoff'] == 0) & (df['shots'] != 0)]
        rmse0 = df['rmse'].min()
        r20 = df['r2'].min()
        rmse.append(rmse0)
        r2.append(r20)
    return rmse, r2

def plot(ax, dataset, min_r2, max_r2, step_r2,  min_rmse, max_rmse, step_rmse):
    rmse,r2 = get_res(dataset)
    ln1 = ax.plot(x,rmse, 'o-', label = r'RMSE')
    ax.yaxis.set_ticks(np.arange(min_rmse, max_rmse, step_rmse))
    ax.set_ylabel(r"RMSE", fontsize = 13)
    ax1 = ax.twinx()
    ln2 = ax1.plot(x,r2 , 'o-', color = 'tab:red', label = r'$r^{2}$')
    ax1.set_ylabel(r"$r^{2}$", fontsize = 13)
    ax1.yaxis.set_ticks(np.arange(min_r2, max_r2, step_r2))
    lns = ln1+ln2
    if dataset == 'lipo':
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc=0)
    if dataset == 'delaney':
        ax.set_xlabel('Dropout rate', fontsize = 13)
    return ax




fig,ax = plt.subplots(3,1, figsize=(7,8))
ax[0] = plot(ax[0], 'lipo', 0.73, 0.75, 0.01,  0.6, 0.8, 0.1)
ax[1] = plot(ax[1], 'freesolv', 0.85, 0.93, 0.03,  0.9, 1.3, 0.1)
ax[2] = plot(ax[2], 'delaney', 0.80, 0.87, 0.02,  0.70, 1, 0.1)

plt.tight_layout()
plt.savefig('dropout.svg')
plt.clf()
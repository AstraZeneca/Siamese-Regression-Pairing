import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.rcParams.update({'font.size': 13})
x = list(range(1,21))
figpath = '../results/MLP_delta/top 1/'
datasets = ['lipo', 'freesolv', 'delaney']

def get_shot_result(dataset):
    path = '../results/MLP_delta/top 1/' + dataset +'/tables/'
    df = pd.read_csv(path + 'avg_res.csv')
    df = df[df['cutoff'] == 0][1:]
    rmse_ls = list(df['rmse'])
    r2_ls = list(df['r2'])
    return rmse_ls
lipo_y = get_shot_result('lipo')
freesolv_y = get_shot_result('freesolv')
delaney_y = get_shot_result('delaney')
min_lipo = min(lipo_y)
min_freesolv = min(freesolv_y)
min_delaney = min(delaney_y)


print(min_lipo)
print(min_freesolv)
print(min_delaney)


relative_lipo = [x - min_lipo for x in lipo_y]
relative_freesolv = [x - min_freesolv for x in freesolv_y]
relative_delaney = [x - min_delaney for x in delaney_y]

#plt.style.use(["paper","single-column"])
plt.plot(x, relative_lipo, 'o-', label = 'Lipophilicity')
plt.plot(x, relative_freesolv, 's-', label = 'Freesolv')
plt.plot(x, relative_delaney, '^-', label = 'ESOL')
plt.ylim(-0.02, 0.25)
plt.yticks(np.arange(0, 0.26, 0.1))

plt.xlabel('Number of Reference Compound', fontsize = 13)
plt.ylabel('RMSE', fontsize = 13)
plt.xticks(np.arange(0,21, 5))
plt.legend(loc = 'upper left')
plt.tight_layout()
plt.savefig(figpath + 'few_shot_plot_relative.svg')
plt.clf()

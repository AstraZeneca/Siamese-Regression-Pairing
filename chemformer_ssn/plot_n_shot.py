import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams.update({'font.size': 12})

def get_shot_res(drp, data):
    path = f'../results/chemformer_snn/dropout{str(drp)}/{data}/tables/'
    df = pd.read_csv(path + 'avg_res.csv')
    df = df[df['cutoff'] == 0][1:]
    return list(df['rmse']), list(df['r2'])
    

rmse_lipo, r2_lipo = get_shot_res(0.0, 'lipo')
rmse_freesolv, r2_freesolv = get_shot_res(0.0, 'freesolv')
rmse_delaney, r2_delaney = get_shot_res(0.0, 'delaney')
shot = list(range(1,21))


fig, ax = plt.subplots(3, 1, sharex=True, figsize=(6,8))
ln1 = ax[0].plot(shot, rmse_lipo, 'o-', label = 'RMSE')
ax[0].yaxis.set_ticks(np.arange(0.5, 0.71, 0.1))
#ax[0].axhline(y =0.594 , xmin=0 , xmax=3,c="blue",linewidth=1,zorder=0, label = 'Chemformer')
#ax[0].set_title('(a)')
#ax[0].set_ylim([0.3, 0.9])
ax[1].plot(shot, rmse_freesolv, 'o-')
#ax[1].axhline(y = 1.020, xmin=0 , xmax=3,c="blue",linewidth=1,zorder=0)
#ax[1].set_title('(b)')

ax[1].yaxis.set_ticks(np.arange(1.1,1.3 , 0.1))
ax[1].set_ylabel('RMSE', fontsize = 13)
ax[2].set_ylabel('RMSE', fontsize = 13)
ax[0].set_ylabel('RMSE', fontsize = 13)
ax[2].plot(shot, rmse_delaney, 'o-')
#ax[2].axhline(y = 0.589, xmin=0 , xmax=3,c="blue",linewidth=1,zorder=0)
#ax[2].set_title('(c)')
ax[2].yaxis.set_ticks(np.arange(0.7, 0.9, 0.1))
ax[2].set_xlabel('Number of Reference Compounds')
ax1 = ax[0].twinx()
ln2 = ax1.plot(shot, r2_lipo, 'ro-', label = r'$r^{2}$')
#ax1.axhline(y = 0.758, xmin=0 , xmax=3,c="blue",linewidth=1,zorder=0, label = 'Chemformer')
lns = ln1 + ln2
labs = [l.get_label() for l in lns]
ax[0].legend(lns, labs, loc=0)
ax1.set_ylim([0.6, 1])
ax1.set_yticks(np.arange(0.6, 1.05, 0.1))
ax1.set_ylabel(r'r$^{2}$', fontsize = 13)
ax2 = ax[1].twinx()
ax2.plot(shot, r2_freesolv, 'ro-')
ax2.set_ylabel(r'r$^{2}$', fontsize = 13)
#ax2.axhline(y = 0.926, xmin=0 , xmax=3,c="blue",linewidth=1,zorder=0)
ax2.set_ylim([0.6, 1])
ax2.set_yticks(np.arange(0.6, 1.05, 0.1))

ax3 = ax[2].twinx()
ax3.plot(shot, r2_delaney, 'ro-')
#ax3.axhline(y = 0.926, xmin=0 , xmax=3,c="blue",linewidth=1,zorder=0)
ax3.set_ylim([0.6, 1])
ax3.set_yticks(np.arange(0.6, 1.05, 0.1))
ax3.set_ylabel(r'r$^{2}$', fontsize = 13)

plt.tight_layout()
plt.xticks(np.arange(0,21, 5))
plt.savefig('few_shot_plot_relative.svg')
plt.clf()

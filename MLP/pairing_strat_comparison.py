import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.rcParams.update({'font.size': 13})
path_head = '../results/MLP_delta/'
datasets = [ 'lipo/', 'freesolv/', 'delaney/']
top_1 = []
top_1_sd = []
half = []
half_sd = []
for i in datasets:
    path_top1 = path_head + 'top 1/' + i + 'tables/'
    df1 = pd.read_csv(path_top1+'avg_res.csv')
    df1 = df1[df1['cutoff'] == 0]
    top_1_idx = df1['rmse'].idxmin()
    top_1.append(df1['rmse'][top_1_idx])
    top_1_sd.append(df1['sd'][top_1_idx])

    path_all = path_head + 'top 0/' + i + 'tables/'
    dfall = pd.read_csv(path_all+'avg_res.csv')
    dfall = dfall[dfall['cutoff'] == 0]
    all_idx = dfall['rmse'].idxmin()
    half.append(dfall['rmse'][all_idx])
    half_sd.append(dfall['sd'][all_idx])


    

# set width of bar
barWidth = 0.2
fig = plt.subplots()
 
 
# Set position of bar on X axis
br1 = np.arange(len(top_1))
br2 = [x + barWidth for x in br1]

 
# Make the plot
plt.bar(br1, top_1, yerr = top_1_sd, width = barWidth,
        edgecolor ='black', label ='Similarity-based pairing', capsize=5)
plt.bar(br2, half, yerr = half_sd, width = barWidth,
        edgecolor ='black', color = "#d62728", label ='Exhaustive pairing', capsize=5)

plt.ylim(0.0, 2.1)
# Adding Xticks
#plt.xlabel('Dataset', fontweight ='bold', fontsize = 15)
plt.ylabel('RMSE', fontsize = 13)
plt.xticks([r + barWidth/2 for r in range(len(top_1))],
        ['Lipophilicity', 'Freesolv', 'ESOL'])
 
plt.legend(loc = 2, fontsize=11)
plt.tight_layout()
plt.savefig('../results/MLP_delta/pairing_strat.png')
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




def visualize(filename):
    csv = pd.read_csv(filename +'.csv')
    csv = csv[csv['SET'] == 'test']
    x = csv['prop']
    plt.hist(x, bins = 20)
    xname = 'prop'
    #xname = csv.columns.values[-1]
    plt.xlabel(xname)
    #plt.xticks(np.arange(min(x), max(x)+1, 2))
    plt.savefig(filename+'_hist_visualizer_test')
    plt.clf()


visualize('0')
visualize('1')
visualize('2')
visualize('3')
visualize('4')
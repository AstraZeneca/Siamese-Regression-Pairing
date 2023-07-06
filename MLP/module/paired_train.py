from rdkit.Chem import AllChem
import pandas as pd
import sys
from module.pair_generation_top import Generator
from  module.pair_generation_all import Generator as Generator_all
from  module.paired_dataset import Dataset
from torch.utils.data import DataLoader
from rdkit import DataStructs
from rdkit import Chem
import torch.nn as nn
import torch
import tqdm
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.preprocessing import MinMaxScaler
from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn
import numpy
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
def plot(train_set,figpath , k):
    fig, ax = plt.subplots()
    question = train_set[(train_set['Similarity'] == 1) & (train_set['prop'] != 0)]
    density_scatter(train_set['Similarity'],train_set['prop'], s = 3, bins = [30,30], ax = ax)
    increment = 0.2
    intervals = numpy.arange(0, 1.2, 0.2)
    averged_error_per_bin = []
    for element in intervals:
        average = cal_sd(train_set, element, element + increment)
        averged_error_per_bin.append(average)
    nega = [-1 * x for x in averged_error_per_bin]
    ax.plot(intervals, averged_error_per_bin, 'r--', label = r'2$\sigma$')
    ax.plot(intervals, nega, 'r--')
    # setting for freesolv
    plt.ylim([-15,17])


    # setting for esol
    # plt.ylim([-7,7])
    # plt.yticks(np.arange(-6,7,3))
    #plt.xlabel('Tanimoto Similarity', fontsize = 15)
    plt.ylabel(r'Exp. $\Delta$free energy', fontsize = 15)
    divider = make_axes_locatable(ax)
    ax_histx = divider.append_axes("top", '20%', pad="3%", sharex=ax)
    ax_histy = divider.append_axes("right", '20%', pad="3%", sharey=ax)
    ax_histx.tick_params(axis="x", labelbottom=False, direction='in')
    ax_histy.tick_params(axis="y", labelleft=False, direction='in')
    ax_histx.hist(train_set['Similarity'], bins=50,density=True, orientation='vertical') #,zorder=0)
    ax_histy.hist(train_set['prop'], bins=50,density=True, orientation='horizontal') #,zorder=0)

    ax_histy.set_xticks(np.arange(0, 0.31, 0.3))



    ax.set_xlim(-0.1, 1.1)
    ax.xaxis.set_ticks(np.arange(0, 1.1, 0.2), labels = [])
    

    ax.legend()
    plt.tight_layout()
    plt.savefig(figpath + k + 'delta_vs_sim.svg')
    plt.clf()

def cal_sd(df1, sim1, sim2):
    df = df1[(df1['Similarity'] >= sim1) & (df1['Similarity'] <= sim2)]

    sd = df['prop'].std() * 2

    return sd

class paired_trainer():
    def __init__(self, figpath, train_set, test_set, top, lr, patience, factor, min_lr, eps, epochs, val_set = None ,file_name = 'lipo'):
        self.train_set = train_set
        self.test_set = test_set
        self.figpath = figpath
        self.lr = lr
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.eps = eps
        self.epochs = epochs
        self.top = top
        if val_set.empty:
            self.val_set = test_set
        else:
            self.val_set = val_set
        self.file_name = file_name
        self.train_fp = [to_fp_ECFP(smi) for smi in self.train_set['smiles']]
        self.test_fp = [to_fp_ECFP(smi) for smi in self.test_set['smiles']]
        self.val_fp = [to_fp_ECFP(smi) for smi in self.val_set['smiles']]

        if top != 0:
            self.train_pair = Generator(self.file_name, df = self.train_set, top = top).maxoutput
            self.val_pair = Generator(self.file_name, df = self.val_set, top = top).maxoutput
            
            
        else:
            self.train_pair = Generator_all(self.file_name, df = self.train_set).maxoutput
            self.val_pair = Generator_all(self.file_name, df = self.val_set).maxoutput            

        self.train_pair.dropna(inplace = True)
        self.val_pair.dropna(inplace = True)
        self.model = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(128, 1, bias = True)
        )
        if torch.cuda.is_available():
            self.model = self.model.cuda()


    def train(self, k, batch_size, figpath):
        k = str(k)
        if self.top == 1:
            plot(self.train_pair,figpath, k)
        train_set = Dataset(self.train_pair, self.train_fp, self.train_fp)
        val_set = Dataset(self.val_pair, self.val_fp, self.val_fp)
        self.trainloader = DataLoader(train_set , batch_size = batch_size, collate_fn =Dataset.collate_fn)
        self.validloader = DataLoader(val_set , batch_size = batch_size, collate_fn =Dataset.collate_fn)




        # Loss and optimizers
        lr = self.lr
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=self.patience,factor=self.factor,min_lr = self.min_lr, eps = self.eps)
        epochs = self.epochs

        train_loss, valid_loss = self.full_gd(criterion, optimizer, scheduler, epochs)

        plt.figure(figsize = (10,8))
        plt.plot(train_loss)
        plt.plot(valid_loss)

        plt.xticks(np.arange(0,epochs, 10))
        plt.savefig(figpath +self.file_name + k +'train_losses_delta.svg')
        plt.clf()


    def full_gd(self, criterion, optimizer, scheduler, epochs):
        # Stuff to store
        model = self.model

        train_loss = np.zeros(epochs)
        valid_loss = np.zeros(epochs)
        min_valid_loss = np.inf
        model.train() 
        for it in range(epochs):
            #for param_group in optimizer.param_groups:
                 # print('LR: ',param_group['lr']) 
            trainloss = 0
            validloss = 0
            count =0
            for i, batch in enumerate(self.trainloader):
                X,Y = batch
                if torch.cuda.is_available():
                    X, Y = X.cuda(), Y.cuda()
                optimizer.zero_grad()
                target = model(X)
                loss = criterion(target,Y)
                trainloss += loss.item()

                loss.backward()
                optimizer.step()
                scheduler.step(loss)
                #print(f'{count}: {loss.item():.12f}')


            train_loss[it] = trainloss/len(self.trainloader)
            model.eval()
            for X, Y in self.validloader:
                if torch.cuda.is_available():
                    X, Y = X.cuda(), Y.cuda()
                with torch.no_grad():
                    target = model(X)
                    loss = criterion(target,Y)
                    validloss += loss.item() 

            valid_loss[it] = validloss/len(self.validloader)
            model.train()
            #print(f'Epoch {it+1} \t\t Training Loss: {train_loss[it]} \t\t Validation Loss: {valid_loss[it]}')
            if min_valid_loss > valid_loss[it]:
                #print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss[it]:.6f}) \t Saving The Model')
                min_valid_loss = valid_loss[it]
                # Saving State Dict
                torch.save(model.state_dict(), self.figpath + 'saved_model.pth')

        
        return train_loss, valid_loss

    def test_train_pair(self):
        df = pd.DataFrame(columns = ['ID1', 'ID2','comp1','comp2', 'sim', 'comprop2','prop'])
        for i in range(len(self.test_set)):
            temp = pd.DataFrame(columns = ['ID1','comp1','comp2', 'sim', 'comprop2','prop'])
            temp['ID1'] = [i] *len(self.train_set)
            temp['ID2'] = self.train_set.iloc[:, 0]
            temp['comp1'] = [self.test_set.iloc[i, -2]] * len(self.train_set)
            temp['comp2'] = self.train_set.iloc[:, -2]
            temp['sim'] = [DataStructs.TanimotoSimilarity(self.test_fp[i] , x) for x in self.train_fp]
            temp['comprop2'] = self.train_set.iloc[:, -1]
            temp['prop'] = [self.test_set.iloc[i,-1] - x for x in temp['comprop2']]
            df = pd.concat([df, temp], ignore_index=True)

        return df



            

    def predict_test(self):
        model = self.model
        model.load_state_dict(torch.load(self.figpath + 'saved_model.pth'))
        model.eval()
        test_train_pair = self.test_train_pair()
        testloader = Dataset(test_train_pair, self.test_fp, self.train_fp)
        testloader = DataLoader(testloader , batch_size = 256, collate_fn =Dataset.collate_fn)
        pred_prop = []
        for i, batch in enumerate(progress_bar(testloader, total=len(testloader))):
            X, Y  = batch
            if torch.cuda.is_available():
                X, Y = X.cuda(), Y.cuda()            
            with torch.no_grad():
                Y_pred = self.model(X)
            Y_pred = [y.cpu().detach().numpy()[0] for y in Y_pred]
            pred_prop+=Y_pred
        test_train_pair['pred_prop'] = pred_prop
        return test_train_pair
        


    

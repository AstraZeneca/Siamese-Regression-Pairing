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


class snn(nn.Module):
    def __init__(self):
        super(snn, self).__init__()

        ### diemension problem
        self.arm = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
        )
        self.regression = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1, bias = True)
        )
    def forward(self, x1,x2):
        # compound 1
        out1 = self.arm(x1)
        # compound 2
        out2 = self.arm(x2)
        x = out1 - out2
        model_output = self.regression(x)

        return model_output






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
        self.model = snn()
        if torch.cuda.is_available():
            self.arm = self.arm.cuda()
            self.regression = self.regression.cuda()



    def train(self, k, batch_size, figpath):
        k = str(k)
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
                X1, X2, Y = batch
                if torch.cuda.is_available():
                    X1, X2, Y = X1.cuda(), X2.cuda(), Y.cuda()
                optimizer.zero_grad()
                target = model(X1, X2)
                loss = criterion(target,Y)
                trainloss += loss.item()

                loss.backward()
                optimizer.step()
                scheduler.step(loss)
                #print(f'{count}: {loss.item():.12f}')


            train_loss[it] = trainloss/len(self.trainloader)
            model.eval()
            for X1, X2, Y in self.validloader:
                if torch.cuda.is_available():
                    X1, X2, Y = X1.cuda(), X2.cuda(), Y.cuda()
                with torch.no_grad():
                    target = model(X1, X2)
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
            X1, X2, Y  = batch
            if torch.cuda.is_available():
                X1, X2, Y = X1.cuda(), X2.cuda(), Y.cuda()            
            with torch.no_grad():
                Y_pred = self.model(X1, X2)
            Y_pred = [y.cpu().detach().numpy()[0] for y in Y_pred]
            pred_prop+=Y_pred
        test_train_pair['pred_prop'] = pred_prop
        return test_train_pair
        


    

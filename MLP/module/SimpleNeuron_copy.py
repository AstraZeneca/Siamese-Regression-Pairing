import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit.Chem import AllChem
import pandas as pd
from rdkit import DataStructs
from rdkit import Chem
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import os
from torch.utils.data import DataLoader
from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn
import numpy
from matplotlib import pylab
from sklearn.metrics import r2_score
def to_fp_ECFP(smi):
    if smi:
        try:
            mol = Chem.MolFromSmiles(smi)
        except Exception:
            print(smi)
        if mol is None:
            return None
        return AllChem.GetMorganFingerprint(mol, 2)

def tanimoto_similarity(smi1, smi2):
    fp1, fp2 = None, None
    fp1 = to_fp_ECFP(smi1)
    fp2 = to_fp_ECFP(smi2)
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def select(train_smi_ls, test_df, cutoff_ls):
    max_sim = []
    rmse_cut_off = []
    r2_cutoff = []
    for i in test_df['smiles']:
        sim_ls = [tanimoto_similarity(i, smi) for smi in train_smi_ls]
        max_sim.append(max(sim_ls))
    test_df['sim'] = max_sim
    for i in cutoff_ls:
        slc = test_df[test_df['sim'] >= i]
        error_squre = slc['error'] **2
        tot = error_squre.sum()/len(error_squre)
        RMSE = math.sqrt(tot)
        r2 = r2_score(slc['real'], slc['pred'])
        r2_cutoff.append(r2)
        rmse_cut_off.append(RMSE)
    return rmse_cut_off, r2_cutoff
class PropDataset(object):
    """The training table dataset.
    """
    def __init__(self, df):
        self.df = df
        X = []
        for fp in df['fp']:
            array = np.zeros((0, ), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, array)
            X.append(array)
        X = np.array(X)
        X = torch.from_numpy(X.astype(np.float32))
        Y = np.array(df.iloc[:,2])
        Y = torch.from_numpy(Y.astype(np.float32).reshape(-1, 1))        
        self.x_data = X # Load the images into torch tensors
        self.y_data = Y # Class labels
        self.len = len(self.x_data) # Size of data
        
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
        
    def __len__(self):
        return self.len

def set_seed(seed = 42):
    """
    Enables reproducibility.
    
    """
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED']=str(seed)

def prepare(df1):
    df = df1.copy()
    df['mol'] = df['smiles'].map(lambda x: Chem.MolFromSmiles(x))
    df.columns.values[0] = 'ID'
    df['fp'] = [AllChem.GetHashedMorganFingerprint(mol,2,2048) for mol in df['mol']]
    return df



def train_classic(k, figpath_classic, train_set, test_set, lr, patience, factor, min_lr, eps, epochs,val_set = None, file_name = 'lipo'):
    k = str(k)
    set_seed(seed = 42)
    lr = lr
    epochs = epochs
    train_set_org = prepare(train_set)
    test_set_org = prepare(test_set)
    if val_set.empty:
        val_set_org = prepare(test_set)
    else:
        val_set_org = prepare(val_set)
    train_set = PropDataset(train_set_org)
    val_set = PropDataset(val_set_org)
    trainloader = DataLoader(train_set, batch_size=32)
    validloader = DataLoader(val_set, batch_size=32)



    model = nn.Sequential(
        nn.Linear(2048, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    )

    if torch.cuda.is_available():
        model = model.cuda()

    # Loss and optimizers
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience = patience,factor=factor,min_lr = min_lr, eps = eps)



    # Train the model
    def full_gd(model, criterion, optimizer, trainloader, validloader, epochs=epochs):
    # Stuff to store
        train_loss = np.zeros(epochs)
        valid_loss = np.zeros(epochs)
        min_valid_loss = np.inf
        train_pred = []
        train_actual = []
        val_pred = []
        val_actual = []
        model.train() 
        for it in range(epochs):
            trainloss = 0
            validloss = 0
            for X, Y in trainloader:
                if torch.cuda.is_available():
                    X, Y = X.cuda(), Y.cuda()
                optimizer.zero_grad()
                target = model(X)
                loss = criterion(target,Y)
                trainloss += loss.item()
                loss.backward()
                optimizer.step()
                scheduler.step(loss)
            train_loss[it] = trainloss/len(trainloader)
            model.eval()
            for X, Y in validloader:
                if torch.cuda.is_available():
                    X, Y = X.cuda(), Y.cuda()
                target = model(X)
                loss = criterion(target,Y)
                validloss += loss.item()        
            valid_loss[it] = validloss/len(validloader)
            model.train()
            print(f'Epoch {it+1} \t\t Training Loss: {train_loss[it]} \t\t Validation Loss: {valid_loss[it]}')
            if min_valid_loss > valid_loss[it]:
                print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss[it]:.6f}) \t Saving The Model')
                min_valid_loss = valid_loss[it]
                # Saving State Dict
                torch.save(model.state_dict(), figpath_classic + 'saved_model.pth')

        
        return train_loss, valid_loss



    train_loss,valid_loss = full_gd(model, criterion, optimizer, trainloader, validloader)
    # calculate RMSE
    RMSE_train = math.sqrt(train_loss[-1])
    #print(RMSE_train)
    RMSE_val = math.sqrt(valid_loss[-1])
    #print(RMSE_val)

    def predict(dataset):
        actual = dataset.iloc[:,2]
        X = []
        for fp in dataset['fp']:
            array = np.zeros((0, ), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, array)
            X.append(array)
        X = np.array(X)
        Y = np.array(actual)
        X = torch.from_numpy(X.astype(np.float32))
        Y = torch.from_numpy(Y.astype(np.float32).reshape(-1, 1))
        if torch.cuda.is_available():
            X, Y = X.cuda(), Y.cuda()
        model.load_state_dict(torch.load(figpath_classic + 'saved_model.pth'))
        pred = model(X)
        loss = criterion(pred, Y)
        return actual, pred, loss




    # move data from GPU to CPU and convert Tensor to numpy array
    train_actual, train_pred, _ = predict(train_set_org)
    val_actual, val_pred, _ = predict(val_set_org)
    test_actual, test_pred,_ = predict(test_set_org)
    val_pred =[x.cpu().detach().numpy()[0] for x in val_pred]
    train_pred =[x.cpu().detach().numpy()[0] for x in train_pred]
    test_pred = [x.cpu().detach().numpy()[0] for x in test_pred]

    def r2(actual, pred):
        yhat = np.sum(actual)/len(pred)
        sstot = np.sum([(x- yhat)**2 for x in actual])
        ls = [x1 - x2 for (x1,x2) in zip(actual, pred)]
        ssres = sum([x**2 for x in ls])
        r_square = 1 - ssres/sstot
        return r_square

    r2_train = r2(train_actual, train_pred)
    #print(r2_train)
    r2_val = r2(val_actual, val_pred)
    r2_test = r2(test_actual, test_pred)
    #text = '''RMSE_train: {0}, RMSE_val: {1} \nR2_train: {2}, R2_val: {3}'''.format(RMSE_train, RMSE_val, r2_train, r2_val)
    #print(text)
    # density plot

    test_df = pd.DataFrame()
    test_df['smiles'] = test_set['smiles']
    test_df['real'] = test_actual
    test_df['pred'] = test_pred
    test_df['error'] = test_df['pred'] - test_df['real']
    cutoff_ls = [0,0.3,0.35,0.40,0.45,0.50]
    rmse_ls, r2_ls = select(train_set_org['smiles'], test_df, cutoff_ls)
    # Plot Train Loss and Validation loss
    plt.figure(figsize = (10,8))
    plt.plot(train_loss)
    plt.plot(valid_loss)
    plt.xticks(np.arange(0,epochs, 10))
    plt.xlabel('Epochs')

    plt.savefig(figpath_classic + k + 'train_losses_single.png')
    plt.clf()
    #print(outputs)
    print('--- preds vs truth ---')
    print('test loss is {}'.format(RMSE_val))

    # plot val actual vs pred
    plt.scatter(val_actual, val_pred, s = 10)
    plt.axline((0, 0), (1, 1), color='k')
    plt.xlabel('Experiment ' + file_name)
    plt.ylabel('Predicted ' + file_name)
    plt.savefig(figpath_classic + k + 'val_actualvspred.png')
    plt.clf()
    print('finish val actual vs pred plot')


    return  rmse_ls, r2_ls






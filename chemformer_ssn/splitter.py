import math
import numpy as np
import pandas as pd
from rdkit.Chem import AllChem
from rdkit import Chem


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


class split(object):
    def __init__(self, DataFrame, train_percentage, test_percentage):
        self.DataFrame = DataFrame
        self.train_percentage = train_percentage
        self.test_percentage = test_percentage
        self.val_percentage = (100 - self.train_percentage - self.test_percentage)

    def index_selector(self):
        """
        Helper function. Selects indices for train and test sets according to given percentages.
        
        : DataFrame (pd.DataFrame):
        
        """
        DataFrame = self.DataFrame
        idx_train = math.ceil(len(DataFrame) * (self.train_percentage/100))
        idx_test = math.ceil(len(DataFrame) * (self.test_percentage/100))
        
        return idx_train, idx_test

    def split(self):
        """
        Random split. Performed in regression tasks.
        
        """
        
        df = self.DataFrame.reset_index(drop = True)
        
        idx_train, idx_test = self.index_selector()
        train_set, test_set, val_set = self.distribute_indexes(df, idx_train, idx_test)
        
        return train_set, test_set, val_set


    def random_split(self):
        """
        Splits the DataFrame into train, test and validation set at random.
        
        : is_regression (bool): regulates whether it is a regression or a classification task. 

        """
        train_set, test_set, val_set = self.split()
        
        return train_set, test_set, val_set

    def distribute_indexes(self, DataFrame, idx_train, idx_test):
        """
        Helper function. Slices original DataFrame into train, test and validation indices.
        
        : DataFrame (pd.DataFrame): contains all data instances.
        : idx_train ():
        : idx_test ():
        
        """
        
        X_train = DataFrame.loc[:idx_train,:].reset_index(drop = True)
        X_test = DataFrame.loc[idx_train:idx_train + idx_test,:].reset_index(drop = True)
        X_val = DataFrame.loc[idx_train + idx_test:,:].reset_index(drop = True)
        
        return X_train, X_test, X_val


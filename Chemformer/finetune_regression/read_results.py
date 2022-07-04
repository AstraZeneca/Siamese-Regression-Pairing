import pandas as pd


def read(name):
    df = pd.read_csv(name)
    RSME = df['RMSE_test'].mean()
    SD_RMSE =  df['RMSE_test'].std()
    R2 = df['R2_test'].mean()
    SD_R2 = df['R2_test'].std()
    print('RMSE is {}, SD of RMSE is {}\nR2 is {}, SD of R2 is {}'.format(RSME, SD_RMSE, R2, SD_R2))

read('liporesults.csv')
read('freesolvresults.csv')
read('delaneyresults.csv')
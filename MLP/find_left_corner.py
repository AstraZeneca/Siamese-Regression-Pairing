
import pandas as pd



path = f'../results/MLP_delta/top 1/lipo/tables'
df = pd.read_csv(f'{path}/0.csv')
df. tail(-1)

sim = df['sim']
err = df['pred_prop'] - df['prop']
df['err'] = err
#
points = df[(df['sim'] <= 0.2) & (df['err'] > 5)]

points.to_csv('question_group.csv')

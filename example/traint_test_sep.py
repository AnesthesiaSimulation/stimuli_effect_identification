import pandas as pd
import numpy as np

np.random.seed(42)

data_measure = pd.read_csv('data/database.csv')

case_id_list = data_measure['caseid'].unique()

np.random.shuffle(case_id_list)

proportion_train = 0.7
nb_caseid_train = int(0.7*len(case_id_list))

train_caseid = case_id_list[:nb_caseid_train]

data_measure['split'] = 'test'

data_measure.loc[
    data_measure['caseid'].isin(train_caseid),
    'split'
] = 'train'


data_measure.to_csv('data/database.csv')
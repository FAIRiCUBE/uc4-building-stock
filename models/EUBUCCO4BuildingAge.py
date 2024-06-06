# -.-|m { input: false }

# standard libs
import os
import sys
import logging

# project lib
PROJECT_SRC_PATH = os.path.join(os.path.abspath(''), '..', 'lib/eubucco')
sys.path.append(PROJECT_SRC_PATH)
PROJECT_SRC_PATH = os.path.join(os.path.abspath(''), '..', 'lib')
sys.path.append(PROJECT_SRC_PATH)
print(PROJECT_SRC_PATH)
import visualizations
from prediction_age import AgePredictor, AgeClassifier, AgePredictorComparison
import preprocessing as pp
from measurer import Measurer

# external libs
import numpy as np
import pandas as pd
import geopandas as gpd
from xgboost import XGBRegressor, XGBClassifier

import pickle

data_path = '/'
measurer = Measurer()
tracker = measurer.start(data_path=data_path)
DATA_DIR = '.'
shape = []

path_data_ESP = r"../data/df-ESP.pkl"

#os.path.join(DATA_DIR, 'df-ESP-exp.pkl')

df = pd.read_pickle(path_data_ESP)
sample_size = int(0.10 * len(df))

# Randomly select 10% of the rows
#df = df.sample(n=sample_size, random_state=42)
df= df[df['city'] == 'Barcelona']
print('LENGHT: ', len(df))


xgb_model_params = {'tree_method': 'hist'}
xgb_hyperparams = {
    'max_depth': 5,
    'learning_rate': 0.1,
    'n_estimators': 500,
    'colsample_bytree': 0.5,
    'subsample': 1.0,
}

predictor = AgePredictor(
    model=XGBRegressor(**xgb_model_params),
    df=df,
    test_training_split=pp.split_80_20,
    # cross_validation_split=pp.cross_validation,
    early_stopping=True,
    hyperparameters=xgb_hyperparams,
    preprocessing_stages=[pp.remove_outliers]
)

predictor.evaluate()
measurer.end(tracker=tracker,
                 shape=shape,
                 libraries=[v.__name__ for k, v in globals().items() if type(v) is ModuleType and not k.startswith('__')],
                 data_path=data_path,
                 program_path=__file__,
                 variables=locals(),
                 csv_file='BuildingAgeRegressor.csv')



measurer = Measurer()
tracker = measurer.start(data_path=data_path)

tabula_nl_bins = [1900, 1965, 1975, 1992, 2006, 2015, 2022]
equally_sized_bins = (1900, 2020, 10)

classifier = AgeClassifier(
    model=XGBClassifier(**xgb_model_params),
    df=df,
    test_training_split=pp.split_80_20,
    # cross_validation_split=pp.cross_validation,
    preprocessing_stages=[pp.remove_outliers],
    hyperparameters=xgb_hyperparams,
    mitigate_class_imbalance=True,
    # bin_config=equally_sized_bins,
    bins=tabula_nl_bins,
)
classifier.evaluate()

measurer.end(tracker=tracker,
                 shape=shape,
                 libraries=[v.__name__ for k, v in globals().items() if type(v) is ModuleType and not k.startswith('__')],
                 data_path=data_path,
                 program_path=__file__,
                 variables=locals(),
                 csv_file='BuildingAgeClassifier.csv')
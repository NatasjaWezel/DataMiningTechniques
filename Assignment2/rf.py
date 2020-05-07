from DM_UTD.DataMiningTechniques.Assignment2.helpers.skprog.wrappers import TreesProgressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

import pandas as pd
import random
import csv
import os
import math
import numpy as np
from tqdm import tqdm
from numba import jit
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

pd.options.mode.chained_assignment = None

# import data
df_train = pd.read_hdf("DM_UTD/DataMiningTechniques/Assignment2/data/traindf_clean.hdf")
df_test = pd.read_hdf("DM_UTD/DataMiningTechniques/Assignment2/data/testdf_clean.hdf")

# convert to x & y
y_train = df_train["importance"]
X_train = df_train.drop(["position","importance","click_bool","booking_bool"], axis=1).copy()
X_test = df_test

# run randomforest with a progress bar (=TreesProgressor)
rfc = TreesProgressor(RandomForestClassifier())
print("DID RFC")

# fit model
model = rfc.fit(X_train, y_train)
print("DID MODEL")

# make predictions
predictions = model.predict(X_test)
print("DID PREDICTIONS")

# save predictions into dataframe and save this as a hdf file
X_test['predicted_importance'] = predictions
X_test.to_hdf("DM_UTD/DataMiningTechniques/Assignment2/data/random_forest_result.hdf", key="df", format="table")

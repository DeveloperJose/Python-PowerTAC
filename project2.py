# Author: Jose G. Perez <josegperez@mail.com>
# PowerTAC Net Demand Predictor for the system
import csv
import os
from timeit import default_timer as timer

import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from project1 import read_data, print_rf_importances

# %% Settings / Configuration
np.set_printoptions(linewidth=np.inf, suppress=True)
np.random.seed(1738)
TRAINING_FOLDER = 'trainingp2'
TESTING_FOLDER = 'testingp2'
MODEL_FILENAME = 'random_forest10_p2.pkl'


def print_timestep(x_data,y_data):
    year = x_data[0]
    month = x_data[1]
    day_date = x_data[2]
    hour = x_data[3]
    day_of_week = x_data[4]

    s = '[Year={},Month={},DayDate={},Hour={},DayWeek={},Demand={}]'.format(
        year, month, day_date, hour, day_of_week, y_data)
    print(s)


# %% Loading files
start_time_load_files = timer()

print('Loading training and testing data')
data_train, games_train = read_data(TRAINING_FOLDER)
data_test, games_test = read_data(TESTING_FOLDER)

data_train = data_train.astype(np.float32)
data_test = data_test.astype(np.float32)

print("== Took %.3f" % (timer() - start_time_load_files), 'sec')

# %% Encoding
# Project 2: 35 features and then the NetDemand (36 columns)
x_train = data_train[:, :35]
y_train = data_train[:, 35]

x_test = data_test[:, :35]
y_test = data_test[:, 35]

# %% Random forest training
if os.path.exists(MODEL_FILENAME):
    start_time_load_rf = timer()

    print('Loading random forest from file')
    rf = joblib.load(MODEL_FILENAME)
    rf.verbose = 0

    print("== Loading RF Took %.3f" % (timer() - start_time_load_rf), 'sec')
else:
    start_time_rf_train = timer()

    print("Training random forest as file was not found")
    rf = RandomForestRegressor(n_estimators=10, verbose=2, n_jobs=-1)
    rf.fit(x_train, y_train)
    rf.verbose = 0

    print("== Training RF Took %.3f" % (timer() - start_time_rf_train), 'sec')

    print("Saving RF model to a file")
    joblib.dump(rf, MODEL_FILENAME)

# %% Stats
feature_list = ['Year', 'Month', 'DayDate', 'Hour', 'DayofWeek',
                'cCloudCov', 'cTemp', 'cWindDir', 'cWindSpd', 'cTotalCons', 'cTotalProd',
                'c1CloudCov', 'c1Temp', 'c1WindDir', 'c1WindSpd', 'c1TotalCons', 'c1TotalProd',
                '24CloudCov', '24Temp', '24WindDir', '24WindSpd', '24TotalCons', '24TotalProd',
                '168CloudCov', '168Temp', '168WindDir', '168WindSpd', '168TotalCons', '168TotalProd',
                'c2_168CloudCov', 'c2_168Temp', 'c2_168WindDir', 'c2_168WindSpd', 'c2_168TotalCons', 'c2_168TotalProd']
print_rf_importances(rf, feature_list)

# %% Prediction (model testing)
preds_baseline = []  # One week back (same hour, same day) strategy
x_rf = []  # Store all Xs to be predicted by the random forest to save computation time by doing it all at once
y_true = []

start_time_testing = timer()
for idx, (x, y) in enumerate(zip(x_test, y_test)):
    # Skip first week
    if idx < 24:
        continue

    prev_week_idx = idx - 24

    assert x[3] == x_test[prev_week_idx][3], 'Not the same hour in the previous week'

    preds_baseline.append(y_test[prev_week_idx])
    x_rf.append(x)
    y_true.append(y)

start_time_rf_pred = timer()

print("Running random forest prediction on all games (to save time)")
rf.verbose = 2
preds_rf = rf.predict(x_rf)
rf.verbose = 0

print("= Random Forest Prediction took %.3f" % (timer() - start_time_rf_pred), 'sec')
print("== Processing testing set took %.3f" % (timer() - start_time_testing), 'sec')

# %% Metrics
metric = metrics.mean_absolute_error
print('[Project 2] Metric {} | OneWeek Baseline {:.5f}, RandomForest {:.5f}'.format(
    metric.__name__,
    metric(y_true, preds_baseline),
    metric(y_true, preds_rf)))
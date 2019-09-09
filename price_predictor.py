# Author: Jose G. Perez <josegperez@mail.com>
# PowerTAC Price Predictor
import csv
import os
from timeit import default_timer as timer

import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from project1 import read_data, print_rf_importances

# %% Settings / Configuration
np.set_printoptions(linewidth=np.inf, suppress=True)
np.random.seed(1738)

# 80 / 20 split
TRAINING_FOLDER = 'training_pricepredictor'
TESTING_FOLDER = 'testing_pricepredictor'
MODEL_FILENAME = 'random_forest10_pricepred.pkl'
BOOST_LINEAR_FILENAME = 'boost.pkl'
BOOST_RF_FILENAME = 'boost_rf.pkl'

# %% Loading files
start_time_load_files = timer()

print('Loading training and testing data')
data_train, games_train = read_data(TRAINING_FOLDER)
data_test, games_test = read_data(TESTING_FOLDER)

data_train = data_train.astype(np.float32)
data_test = data_test.astype(np.float32)

print("== Took %.3f" % (timer() - start_time_load_files), 'sec')

# %% Encoding
# Project 3: 18 features and then the clearing price
x_train = data_train[:, :-1]
y_train = data_train[:, -1]

x_test = data_test[:, :-1]
y_test = data_test[:, -1]

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

# %% Prediction
# Baseline = 0.7 * PrevWeekSameDaySameHourSameHourAhead (f12) + 0.3 * CurWeekPrevHour (f10)
print("Running baseline prediction")
y_baseline = (0.7 * x_test[:, 11]) + (0.3 * x_test[:, 9])


# Explore different predictor models for stacking (error fixing), logistic regression, random forests
print("Running random forest prediction")
start_time_rf_pred = timer()
rf.verbose = 2
y_rf_pred = rf.predict(x_test)
rf.verbose = 0

print("= Random Forest Prediction took %.3f" % (timer() - start_time_rf_pred), 'sec')

# %% Metrics
metric = metrics.mean_absolute_error
print('[Price Predictor] Metric {} | RandomForest {:.5f} | Baseline {:.5f}'.format(
    metric.__name__,
    metric(y_test, y_rf_pred),
    metric(y_test, y_baseline)
))

# %% Create dataset for gradient boosting
print("Creating dataset for gradient boosting")
start_time_generate_boost_data = timer()

x_boost = y_rf_pred
y_boost = np.array([y_true - y_pred for (y_true,y_pred) in zip(y_test, y_rf_pred)])

# No split
x_train_boost = x_boost[0:1370625].reshape(-1, 1)
y_train_boost = y_boost[0:1370625]

x_test_boost = x_boost[0].reshape(-1, 1)
y_test_boost = y_boost[0]

# 80/20 split, reshape because it's a single feature
# x_train_boost = x_boost[0:1370625].reshape(-1, 1)
# y_train_boost = y_boost[0:1370625]
#
# x_test_boost = x_boost[1370625:-1].reshape(-1, 1)
# y_test_boost = y_boost[1370625:-1]

print("= Took %.3f" % (timer() - start_time_generate_boost_data), 'sec')

# %% Boosting model training (linear)
if os.path.exists(BOOST_LINEAR_FILENAME):
    start_time_load_boost = timer()

    print('Loading booster linear model from file')
    boost_ln = joblib.load(BOOST_LINEAR_FILENAME)

    print("== Loading took %.3f" % (timer() - start_time_load_boost), 'sec')
else:
    start_time_boost_train = timer()

    print("Training boosting linear regressor as file was not found")
    boost_ln = LinearRegression()
    boost_ln.fit(x_train_boost, y_train_boost)

    print("== Training took %.3f" % (timer() - start_time_boost_train), 'sec')

    print("Saving linear model to a file")
    joblib.dump(boost_ln, BOOST_LINEAR_FILENAME)

# %% Boosting model training (rf)
if os.path.exists(BOOST_RF_FILENAME):
    start_time_load_boost = timer()

    print('Loading booster rf model from file')
    boost_rf = joblib.load(BOOST_RF_FILENAME)

    print("== Loading took %.3f" % (timer() - start_time_load_boost), 'sec')
else:
    start_time_boost_train = timer()

    print("Training boosting rf regressor as file was not found")
    boost_rf = RandomForestRegressor(n_estimators=10, verbose=2, n_jobs=-1)
    boost_rf.fit(x_train_boost, y_train_boost)

    print("== Training took %.3f" % (timer() - start_time_boost_train), 'sec')

    print("Saving rf model to a file")
    joblib.dump(boost_rf, BOOST_RF_FILENAME)

# %% Boosting model testing
print("Boosting model prediction")
y_boost_ln_pred = boost_ln.predict(x_test_boost)
y_boost_rf_pred = boost_rf.predict(x_test_boost)

print('[Boosting Model] Metric {} | Linear Regressor {:.5f} | Random Forest {:.5f}'.format(
    metric.__name__,
    metric(y_test_boost, y_boost_ln_pred),
    metric(y_test_boost, y_boost_rf_pred)
))

# %% Boosting model applied
print("Applying boost to previous model test set")
h1 = boost_ln.predict(y_rf_pred.reshape(-1, 1)).flatten()
h2 = boost_rf.predict(y_rf_pred.reshape(-1, 1)).flatten()
pred1 = y_rf_pred + h1
pred2 = y_rf_pred + h2
print('[Price Predictor] Metric {} | RandomForest {:.5f} | Baseline {:.5f} | GradientBoost (Linear Regressor) {:.5f} | GradientBoost (RandomForest) {:.5f}'.format(
    metric.__name__,
    metric(y_test, y_rf_pred),
    metric(y_test, y_baseline),
    metric(y_test, pred1),
    metric(y_test, pred2)
))
# Author: Jose G. Perez <josegperez@mail.com>
# PowerTAC Consumer/Producer Demand Predictor
import csv
import os
from timeit import default_timer as timer

import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# %% Settings / Configuration
np.set_printoptions(linewidth=np.inf, suppress=True)
np.random.seed(1738)
TRAINING_FOLDER = 'training'
TESTING_FOLDER = 'testing'
MODEL_FILENAME = 'random_forest10_p1.pkl'


# %% Function declarations
def read_csv(path):
    rows = []
    with open(path, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            rows.append(row)

    return np.array(rows, dtype=np.object)


def read_data(folder):
    data = []
    games = []
    for fname in os.listdir(folder):
        fdata = read_csv(os.path.join(folder, fname))
        data.extend(fdata)
        games.append(fdata)

    return np.array(data), np.array(games)


def print_timestep(x, y, decode=True):
    name = enc_name.inverse_transform(int(x[0])) if decode else x[0]
    power_type = enc_type.inverse_transform(int(x[1])) if decode else x[1]
    population = x[2]
    date = x[3]
    month = x[4]
    day = x[5]
    hour = x[6]
    cloud_coverage = x[7]
    temperature = x[8]
    wind_direction = x[9]
    wind_speed = x[10]
    demand = y

    s = '[Customer={},Type={},Pop={},Date={},Month={},Day={},Hour={},CloudCov={},Temp={},WindDir={},WindSpeed={},Demand={}]'.format(
        name, power_type, population, date, month, day, hour, cloud_coverage, temperature, wind_direction, wind_speed, demand)

    print(s)


enc_name = LabelEncoder()
enc_type = LabelEncoder()
enc = OneHotEncoder(categorical_features=[0, 1])  # Only indices 0 and 1 are categorical


def encode(data):
    data = data.copy()
    # Project 1: 11 features and then the Customer Production/Consumption (12 columns)
    # Project 2: 35 features and then the NetDemand (36 columns)
    x = data[:, :11]
    y = data[:, 11].astype(np.float32)

    encode_name = enc_name.transform if hasattr(enc_name, 'classes_') else enc_name.fit_transform
    encode_type = enc_type.transform if hasattr(enc_name, 'classes_') else enc_type.fit_transform

    x[:, 0] = encode_name(x[:, 0])
    x[:, 1] = encode_type(x[:, 1])

    x = x.astype(np.float32)
    return x, y


# %% Loading files
start_time_load_files = timer()

print('Loading training and testing data')
data_train, games_train = read_data(TRAINING_FOLDER)
data_test, games_test = read_data(TESTING_FOLDER)

print("== Took %.3f" % (timer() - start_time_load_files), 'sec')

# %% Encoding CustomerName and PowerType
# data[CustomerName0,PowerType1,Population2,Date3,Month4,Day5,Hour6,Cloud7,Temp8,WindDir9,WindSpeed10,Net11]
start_time_encoding = timer()

print("Encoding")
x_train, y_train = encode(data_train)
x_test, y_test = encode(data_test)

print("== Encoding Took %.3f" % (timer() - start_time_encoding), 'sec')

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
feature_list = ['CustomerName0', 'PowerType1', 'Population2', 'Date3', 'Month4', 'Day5', 'Hour6', 'Cloud7', 'Temp8', 'WindDir9', 'WindSpeed10', 'Net11']
# pred = rf.predict(x_test)
# errors = abs(pred - y_test)
# print('Mean Absolute Error:', np.mean(errors))

importances = list(rf.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

# %% Perform predictions
preds_one_day = []
preds_one_week = []
y_true = []
x_rf = []  # We will store all the testing X and run the RF prediction only once to save time

start_time_all_games = timer()
for game_idx, game in enumerate(games_test):
    print("Processing game {}/{}".format(game_idx, len(games_test)))
    start_time_game_process = timer()

    game_x, game_y = encode(game)
    raise Exception()
    prev_end = 0
    for cust_idx, (cust_x, cust_y) in enumerate(zip(game_x, game_y)):
        prev_x = game_x[prev_end:cust_idx]
        if len(prev_x) == 0:  # Skip the first customer in the game (no previous day, no previous week)
            continue

        # Flip so we only consider the most recent time steps first when using argmax
        prev_x = np.flip(prev_x, axis=0)

        # Look for the customer name in the previous time steps

        cust_name = cust_x[0]
        prev_cust_names = prev_x[:, 0]
        prev_timestep_idxs = np.nonzero(prev_cust_names == cust_name)[0]
        if len(prev_timestep_idxs) < 168:  # Skip the first week
            continue

        prev_y = np.flip(game_y[0:cust_idx], axis=0)

        # 1 hour back = 1 time step back [index 0]
        # 1 day back = 24 time steps back [index 23]
        # 1 week back = 168 time steps back [index 167]
        prev_day_idx = prev_timestep_idxs[23]
        prev_week_idx = prev_timestep_idxs[167]

        preds_one_day.append(prev_y[prev_day_idx])
        preds_one_week.append(prev_y[prev_week_idx])
        x_rf.append(cust_x)
        y_true.append(cust_y)

    print("== Processing game took %.3f" % (timer() - start_time_game_process), 'sec')

start_time_rf_pred = timer()

print("Running random forest prediction on all games (to save time)")
rf.verbose = 2
preds_rf = rf.predict(x_rf)
rf.verbose = 0

print("= Random Forest Prediction took %.3f" % (timer() - start_time_rf_pred), 'sec')
print("== Processing all games took %.3f" % (timer() - start_time_all_games), 'sec')

# %% Metrics
metric = metrics.mean_absolute_error
print('Metric {} | OneDay {:.5f}, OneWeek {:.5f}, RandomForest {:.5f}'.format(
    metric.__name__,
    metric(y_true, preds_one_day),
    metric(y_true, preds_one_week),
    metric(y_true, preds_rf)))

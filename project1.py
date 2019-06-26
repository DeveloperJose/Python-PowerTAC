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
TEST_CUSTOMER_TYPES = ['CONSUMPTION', 'INTERRUPTIBLE_CONSUMPTION', 'THERMAL_STORAGE_CONSUMPTION', 'ELECTRIC_VEHICLE']
# TEST_CUSTOMER_TYPES = ['CONSUMPTION', 'INTERRUPTIBLE_CONSUMPTION', 'THERMAL_STORAGE_CONSUMPTION', 'ELECTRIC_VEHICLE', 'SOLAR_PRODUCTION', 'WIND_PRODUCTION', 'BATTERY_STORAGE']
# TEST_CUSTOMER_TYPES = ['SOLAR_PRODUCTION', 'WIND_PRODUCTION']


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


def print_rf_importances(rf_model, features):
    assert len(features) == len(rf_model.feature_importances_), 'Feature list size does not match RF feature size'

    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(features, rf_model.feature_importances_)]
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    for pair in feature_importances:
        print('Variable: {:20} Importance: {}'.format(*pair))


enc_name = LabelEncoder()
enc_type = LabelEncoder()
enc = OneHotEncoder(categorical_features=[0, 1])  # Only indices 0 and 1 are categorical


def encode(data):
    data = data.copy()
    # Project 1: 11 features and then the Customer Production/Consumption (12 columns)
    x = data[:, :11]
    y = data[:, 11].astype(np.float32)

    encode_name = enc_name.transform if hasattr(enc_name, 'classes_') else enc_name.fit_transform
    encode_type = enc_type.transform if hasattr(enc_name, 'classes_') else enc_type.fit_transform

    x[:, 0] = encode_name(x[:, 0])
    x[:, 1] = encode_type(x[:, 1])

    x = x.astype(np.float32)
    return x, y


if __name__ == '__main__':
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
    print_rf_importances(rf, feature_list)

    # %% Perform predictions
    preds_one_day = []
    preds_one_week = []
    y_true = []
    x_rf = []  # We will store all the testing X and run the RF prediction only once to save time

    start_time_testing = timer()

    for game_idx, game in enumerate(games_test):
        start_time_process_game = timer()

        print("Processing game {}/{}".format(game_idx, len(games_test)))
        game_x, game_y = encode(game)

        # To speed up computations looking up past week and past day data, split the dataset per hour
        diff = np.where(np.diff(game_x[:, 6]))[0]
        diff = np.insert(diff, 0, values=-1)

        x_byhour = []
        y_byhour = []
        for diff_idx, diff_start in enumerate(diff):
            if diff_idx >= len(diff) - 1:  # Cannot use last two elements for slicing
                break

            diff_end = diff[diff_idx + 1]
            x_byhour.append(game_x[slice(diff_start + 1, diff_end)])
            y_byhour.append(game_y[slice(diff_start + 1, diff_end)])

        for hour_idx, (hour_block_x, hour_block_y) in enumerate(zip(x_byhour, y_byhour)):
            if hour_idx < 168:  # Skip first week (168 hours)
                continue

            prev_week_block_idx = hour_idx - 168
            prev_day_block_idx = hour_idx - 24

            prev_week_block_x = x_byhour[prev_week_block_idx]
            prev_week_block_y = y_byhour[prev_week_block_idx]

            prev_day_block_x = x_byhour[prev_day_block_idx]
            prev_day_block_y = y_byhour[prev_day_block_idx]

            for cust_idx, (cust_x, cust_y) in enumerate(zip(hour_block_x, hour_block_y)):
                # Only consider certain customers (this adds a 15 second overhead per game)
                cust_type = enc_type.inverse_transform(int(cust_x[1]))
                if cust_type not in TEST_CUSTOMER_TYPES:
                    continue

                # Find the customer in the previous week same hour block
                cust_name = cust_x[0]
                prev_week_matches = np.nonzero(prev_week_block_x[:, 0] == cust_name)[0]
                if len(prev_week_matches) != 1:  # The customer was not found in the previous week
                    continue

                prev_day_matches = np.nonzero(prev_day_block_x[:, 0] == cust_name)[0]
                if len(prev_day_matches) != 1:
                    continue

                prev_week_cust_idx = prev_week_matches[0]
                prev_day_cust_idx = prev_day_matches[0]

                assert prev_week_block_x[prev_week_cust_idx][0] == cust_name, 'Not the same customer name previous week'
                assert prev_day_block_x[prev_day_cust_idx][0] == cust_name, 'Not the same customer name previous day'
                assert prev_week_block_x[prev_week_cust_idx][6] == cust_x[6], 'Not the same hour in the previous week'
                assert prev_day_block_x[prev_day_cust_idx][6] == cust_x[6], 'Not the same hour in the previous day'

                preds_one_day.append(prev_day_block_y[prev_day_cust_idx])
                preds_one_week.append(prev_week_block_y[prev_week_cust_idx])
                x_rf.append(cust_x)
                y_true.append(cust_y)
        print("== Processing game took %.3f" % (timer() - start_time_process_game), 'sec')

    start_time_rf_pred = timer()

    print("Running random forest prediction on all games (to save time)")
    rf.verbose = 2
    preds_rf = rf.predict(x_rf)
    rf.verbose = 0

    print("= Random Forest prediction took %.3f" % (timer() - start_time_rf_pred), 'sec')
    print("== Processing testing set took %.3f" % (timer() - start_time_testing), 'sec')

    # %% Metrics
    metric = metrics.mean_absolute_error
    print('[Project 1] Metric {} | OneDay {:.5f}, OneWeek {:.5f}, RandomForest {:.5f}'.format(
        metric.__name__,
        metric(y_true, preds_one_day),
        metric(y_true, preds_one_week),
        metric(y_true, preds_rf)))

    print('Tested Customer Types: ', TEST_CUSTOMER_TYPES)
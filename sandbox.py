import csv
import os
import numpy as np
from sklearn.externals import joblib
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from timeit import default_timer as timer

#%% Settings / Configuration
np.set_printoptions(linewidth=np.inf)
np.random.seed(1738)
TRAINING_FOLDER = 'training'
TESTING_FOLDER = 'testing'
MODEL_FILENAME = 'random_forest_10.pkl'


#%% Function declarations
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


enc_name = LabelEncoder()
enc_type = LabelEncoder()
enc = OneHotEncoder(categorical_features = [0,1]) # Only indices 0 and 1 are categorical
def encode(data):
    # Project 1: 11 features and then the Customer Production/Consumption
    # Project 2: 35 features and then the NetDemand
    if len(data) == 12:
        x = data[:,:11]
        y = data[:,11].astype(np.float32)

        encode_name = enc_name.transform if hasattr(enc_name, 'classes_') else enc_name.fit_transform
        encode_type = enc_type.transform if hasattr(enc_name, 'classes_') else enc_type.fit_transform

        x[:,0] = encode_name(x[:,0])
        x[:,1] = encode_type(x[:,1])
    elif len(data) == 36:
        raise Exception('36 dataset')

    x = x.astype(np.float32)
    return x, y


#%% Loading files
start_time_load_files = timer()

print('Loading training and testing data')
data_train, games_train = read_data(TRAINING_FOLDER)
data_test, games_test = read_data(TESTING_FOLDER)

print("== Took %.3f" % (timer()-start_time_load_files), 'sec')

#%% Encoding CustomerName and PowerType
# data[CustomerName0,PowerType1,Population2,Date3,Month4,Day5,Hour6,Cloud7,Temp8,WindDir9,WindSpeed10,Net11]
start_time_encoding = timer()

print("Encoding")
x_train, y_train = encode(data_train)
x_test, y_test = encode(data_test)

print("== Encoding Took %.3f" % (timer()-start_time_encoding), 'sec')

#%% Random forest training
if os.path.exists(MODEL_FILENAME):
    start_time_load_rf = timer()

    print('Loading random forest from file')
    rf = joblib.load(MODEL_FILENAME)
    rf.verbose=0

    print("== Loading RF Took %.3f" % (timer()-start_time_load_rf), 'sec')
else:
    start_time_rf_train = timer()

    print("Training random forest as file was not found")
    rf = RandomForestRegressor(n_estimators=10,verbose=2,n_jobs=-1)
    rf.fit(x_train, y_train)
    rf.verbose = 0

    print("== Training RF Took %.3f" % (timer()-start_time_rf_train), 'sec')

    print("Saving RF model to a file")
    joblib.dump(rf, MODEL_FILENAME)

#%% Stats
feature_list = ['CustomerName0','PowerType1','Population2','Date3','Month4','Day5','Hour6','Cloud7','Temp8','WindDir9','WindSpeed10','Net11']
# pred = rf.predict(x_test)
# errors = abs(pred - y_test)
# print('Mean Absolute Error:', np.mean(errors))

importances = list(rf.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

#%% Split a game of bootstrap data into the two weeks for testing
preds_one_day = []
preds_one_week = []
y_true = []
x_rf = []  # We will store all the testing X and run the RF prediction only once to save time

start_time_all_games = timer()
for idx, game_t in enumerate(games_test):
    print("Processing game {}/{}".format(idx, len(games_test)))
    start_time_game_process = timer()

    game = game_t.copy()
    game_x, game_y = encode(game)
    customers = game.shape[0] // 359

    # [Bootstrap] 24 hours before the 2 weeks
    LEN_HOUR1 = customers+1  # The first hour is different and has 1 extra customer (only the case for the training data)
    IDX_DAY_START = 0
    IDX_DAY1_END = LEN_HOUR1 + (23 * customers) # The other 23 hours are regular
    g_24hours_x = game_x[:IDX_DAY1_END]
    g_24hours_y = game_y[:IDX_DAY1_END]

    # [Bootstrap] Week #1, 168 hours
    IDX_WEEK1_END = IDX_DAY1_END + (168 * customers)
    g_week1_x = game_x[IDX_DAY1_END:IDX_WEEK1_END]
    g_week1_y = game_y[IDX_DAY1_END:IDX_WEEK1_END]

    # [Bootstrap] Week #2, 168 hours
    g_week2_x = game_x[IDX_WEEK1_END:]
    g_week2_y = game_y[IDX_WEEK1_END:]

    # [Bootstrap] Both weeks, 336 hours
    g_weeks_x = np.vstack((g_week1_x, g_week2_x))
    g_weeks_y = np.hstack((g_week1_y, g_week2_y))

    print("Running predictions on the specified game")
    # print("[Skipping",enc_type.inverse_transform(0),enc_type.inverse_transform(4),enc_type.inverse_transform(6),']', end='')
    for idx_customer in range(customers):
        info = g_week1_x[idx_customer]
        cust_type = int(info[1])
        # if cust_type == 4 or cust_type == 6:
        # print(enc_type.inverse_transform(cust_type), end='')
        days = np.arange(7, 15)
        idx_curr_day = (days * customers) + idx_customer
        idx_prev_day = ((days-1) * customers) + idx_customer
        idx_prev_week = ((days-7) * customers) + idx_customer

        x_rf.extend(g_weeks_x[idx_curr_day])
        y_true.extend(g_weeks_y[idx_curr_day])
        preds_one_day.extend(g_weeks_y[idx_prev_day])
        preds_one_week.extend(g_weeks_y[idx_prev_week])

    print("== Processing game took %.3f" % (timer() - start_time_game_process), 'sec')


start_time_rf_pred = timer()

print("Running random forest prediction on all games (to save time)")
rf.verbose=2
preds_rf = rf.predict(x_rf)
rf.verbose=0

print("= Random Forest Prediction took %.3f" % (timer() - start_time_rf_pred), 'sec')
print("== Processing all games took %.3f" % (timer() - start_time_all_games), 'sec')

#%% Metrics
metric = metrics.mean_absolute_error
print('Metric {} | OneDay {:.5f}, OneWeek {:.5f}, RandomForest {:.5f}'.format(
    metric.__name__,
    metric(y_true, preds_one_day),
    metric(y_true, preds_one_week),
    metric(y_true, preds_rf)))
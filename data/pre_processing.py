import pandas as pd
from sklearn.preprocessing import Normalizer


def static_obstacle_data():
    # FILE_PATH_TRAIN = '/home/mae/ownCloud/Exchange/Shared outside/1920 Cyberprotect/Data/200610/test_record_iter_dynamic.csv'
    FILE_PATH_TRAIN = '/home/mae/git/robacademy/academy/experiments/reach/safety_layer/results/static_obstacle/bnn.csv'

    INPUT_FEATURES = ['action_0', 'action_1', 'action_2',
                      'grip-pos_0', 'grip-pos_1', 'grip-pos_2',
                      'grip-vel_0', 'grip-vel_1', 'grip-vel_2',
                      'obstacle-body-position-2_0', 'obstacle-body-position-2_1',
                      'obstacle-body-position-2_2',
                      'obstacle-body-velocity-2_0',
                      'obstacle-body-velocity-2_1', 'obstacle-body-velocity-2_2',
                      'desired_goal_0',
                      'desired_goal_1', 'desired_goal_2']

    # INPUT_FEATURES = ['grip_pos_x', 'grip_pos_y', 'grip_pos_z',
    #                   'grip_vel_x', 'grip_vel_y', 'grip_vel_z',
    #                   'obst_pos_x_1', 'obst_pos_y_1', 'obst_pos_z_1',
    #                   'obst_vel_x_1', 'obst_vel_y_1', 'obst_vel_z_1',
    #                   'goal_pos_x', 'goal_pos_y', 'goal_pos_z',
    #                   'action_x', 'action_y', 'action_z']

    OUTPUT_FEATURES = ['is_viable']  # ONLY 1 feature is currently supported
    # print(ORIGIN_DATA.loc[:, label].value_counts())
    ORIGIN_DATA = pd.read_csv(FILE_PATH_TRAIN)
    return ORIGIN_DATA, INPUT_FEATURES, OUTPUT_FEATURES


def get_processed_data(origin_data=None, input_features=None, output_features=None, normalize=True):
    if origin_data is None:
        origin_data, input_features, output_features = static_obstacle_data()
    # select the needed data by columns index
    all_features = input_features + output_features
    label = output_features[0]
    origin_data[label] = origin_data[label].astype(int)
    # Make the number of safe_state & unsafe_state equal
    n_safe_state = origin_data.loc[origin_data[label] == True]
    n_unsafe_state = origin_data.loc[origin_data[label] == False]
    n_safe_state = n_safe_state.sample(n=len(n_unsafe_state))  # make the number of safe and unsafe be equal

    prepocessed_data = pd.concat([n_safe_state, n_unsafe_state], axis=0)
    prepocessed_data = prepocessed_data.sample(frac=1)  # select sampled safe data
    prepocessed_data = prepocessed_data.loc[:, all_features]

    #data = prepocessed_data.iloc[0:12000]
    data = prepocessed_data
    data = data.values
    X, Y = data[:, :-1], data[:, -1]
    print(X.shape, Y.shape)

    X_normalizer = None
    if normalize:
        X, X_normalizer = normalize_data(X)

    return X, Y, X_normalizer


def normalize_data(data):
    scaler = Normalizer().fit(data)
    scaled_data = scaler.transform(data)
    return scaled_data, scaler

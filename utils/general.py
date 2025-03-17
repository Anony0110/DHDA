import os
import random
from sklearn.preprocessing import StandardScaler
import numpy as np
import math

import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import _tree
import warnings

warnings.filterwarnings("ignore")


# the function to extract the dividing conditions recursively, and divide the training data into clusters (divisions)
def recursive_dividing(node, depth, tree_, X, samples=[], max_depth=1, min_samples=2, cluster_indexes_all=[]):
    indent = "  " * depth
    if depth <= max_depth:
        if tree_.feature[node] != _tree.TREE_UNDEFINED:  # if it's not the leaf node
            left_samples = []
            right_samples = []
            # get the node and the dividing threshold
            name = tree_.feature[node]
            threshold = tree_.threshold[node]
            # split the samples according to the threshold
            for i_sample in range(0, len(samples)):
                if X[i_sample, name] <= threshold:
                    left_samples.append(samples[i_sample])
                else:
                    right_samples.append(samples[i_sample])
            # check if the minimum number of samples is statisfied
            if (len(left_samples) <= min_samples or len(right_samples) <= min_samples):
                # print('{}Not enough samples to cluster with {} and {} samples'.format(indent,len(left_samples),len(right_samples)))
                cluster_indexes_all.append(samples)
            else:
                # print("{}{} samples with feature {} <= {}:".format(indent, len(left_samples), name, threshold))
                cluster_indexes_all = recursive_dividing(tree_.children_left[node], depth + 1, tree_, X, left_samples,
                                                         max_depth, min_samples, cluster_indexes_all)
                #print("{}{} samples with feature {} > {}:".format(indent, len(right_samples), name, threshold))
                cluster_indexes_all = recursive_dividing(tree_.children_right[node], depth + 1, tree_, X, right_samples,
                                                         max_depth, min_samples, cluster_indexes_all)
        else:
            cluster_indexes_all.append(samples)
    # the base case: add the samples to the cluster
    elif depth == max_depth + 1:
        cluster_indexes_all.append(samples)
    return cluster_indexes_all


def get_non_zero_indexes(whole_data, total_tasks):
    (N, n) = whole_data.shape
    n = n - 1
    delete_index = set()
    temp_index = list(range(N))
    for i in range(total_tasks):
        temp_Y = whole_data[:, n - i]
        for j in range(len(temp_Y)):
            if temp_Y[j] == 0:
                delete_index.add(j)
    non_zero_indexes = np.setdiff1d(temp_index, list(delete_index))
    return non_zero_indexes


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    m = X.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def get_whole_data(path):
    df = pd.read_csv(path)
    ndarray = df.values
    return ndarray


def get_attributes_result(data):
    X = data[:, :-1]
    Y = data[:, -1][:, np.newaxis]
    return X, Y


def generate_unique_random_numbers(n, seed=0):
    # 创建一个包含0到n的数组
    # random.seed(seed)
    numbers = list(range(n + 1))
    # Fisher-Yates shuffle算法
    for i in range(len(numbers) - 1, 0, -1):
        j = random.randint(0, i)
        numbers[i], numbers[j] = numbers[j], numbers[i]
    return numbers


def process_training_data(multi_env_list=[], n_drifts=4, min_occupation=0.6, max_occupation=0.9, max_count=3000,
                          init_train_count=50, isstandarlize=False, reuse_test=False):
    count_list = {}
    true_change_point_x = list()
    env_count = len(multi_env_list)
    selected_list = generate_unique_random_numbers(env_count - 1)
    whole_data = None
    change_time = 0
    init_data = None
    count = 0
    change_point = list()
    if not reuse_test:
        random_list = selected_list
    else:
        selected_list = selected_list[:int(n_drifts / 2)]
        i = 1
        random_list = [selected_list[0]]
        while i < n_drifts:
            env = np.random.choice(selected_list)
            if env != random_list[-1]:
                i += 1
                random_list.append(env)
    for i in random_list:
        if count == n_drifts:
            break

        whole_data_temp = get_whole_data(multi_env_list[i])
        data_num = len(whole_data_temp)
        low = int(data_num * min_occupation)
        high = int(data_num * max_occupation)
        sample_count = random.randint(low, high)
        if sample_count > max_count:
            sample_count = max_count
        dataset_name = multi_env_list[i].split('\\')[-1]
        count_list[count] = [dataset_name, sample_count]
        count += 1

        all_index = np.array(list(range(data_num)))
        # np.random.seed(seed)
        data_index = np.random.choice(all_index, size=sample_count)
        data = whole_data_temp[data_index]
        change_time += 1

        if whole_data is None:
            # whole_data = data
            init_count = len(data)
            init_test_count = init_count - init_train_count
            all_index_temp = np.array(list(range(len(data))))

            train_index_temp = np.random.choice(all_index_temp, size=init_train_count)
            train_data = data[train_index_temp]
            test_index_temp = np.setdiff1d(all_index_temp, train_index_temp)
            test_data = data[test_index_temp]
            whole_data = test_data
            init_data = train_data
        else:
            whole_data = np.vstack((whole_data, data))
        print(
            f"change point is index {len(whole_data) + init_train_count}, change point is {(len(whole_data) + init_train_count) / 32}")
        true_change_point_x.append((len(whole_data) + init_train_count) / 32)
        change_point.append(len(whole_data))
    print(f"change time is {change_time}")
    print("data info")
    print(f"init train data count = {init_train_count}, stream data count = {len(whole_data)}")
    true_change_point_x.pop()
    # print(whole_data)

    x_test, y_test = get_attributes_result(whole_data)
    x_train, y_train = get_attributes_result(init_data)
    if isstandarlize:
        all_data = np.vstack((x_train, x_test))
        scaler = StandardScaler()
        scaler.fit(all_data)
        data = scaler.transform(all_data)
        x_train = data[:init_train_count]
        x_test = data[init_train_count:]

    return {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test,
            'count_list': count_list}


def build_model(regression_mod='RF', test_mode=True, model_reuse=True, training_X=[], training_Y=[]):
    """
    to build the specified regression model, given the training data
    :param regression_mod: the regression model to build
    :param test_mode: won't tune the hyper-parameters if test_mode == False
    :param training_X: the array of training features
    :param training_Y: the array of training label
    :return: the trained model
    """
    model = None
    if regression_mod == 'RF':
        if model_reuse:
            model = RandomForestRegressor(random_state=0)
        else:
            model = RandomForestRegressor(random_state=0)
        max = (len(training_X) >= 5 and 5 or len(training_X))
        if 500 > len(training_X) > 50:
            max = 5
        elif len(training_X) >= 500:
            max = 10
        max_estimators = (len(training_X) >= 500 and 500 or 100)
        param = {'n_estimators': np.arange(10, max_estimators, 5),
                 'criterion': ['squared_error'],
                 'min_samples_leaf': np.arange(1, max, 2)}
        if not test_mode:
            # print('Hyperparameter Tuning...')
            gridS = GridSearchCV(model, param)
            gridS.fit(training_X, training_Y)
            model = RandomForestRegressor(**gridS.best_params_, random_state=0, warm_start=True)

    elif regression_mod == 'DT':
        if model_reuse:
            model = DecisionTreeRegressor(random_state=0)
        else:
            model = DecisionTreeRegressor(random_state=0)
        max = (len(training_X) >= 5 and 5 or len(training_X))
        if 500 > len(training_X) > 50:
            max = 5
        elif len(training_X) >= 500:
            max = 10
        param = {'criterion': ['squared_error'],
                 'splitter': ['best'],
                 'min_samples_leaf': np.arange(1, max, 2)
                 }
        if not test_mode:
            # print('Hyperparameter Tuning...')
            gridS = GridSearchCV(model, param)
            gridS.fit(training_X, training_Y)
            model = DecisionTreeRegressor(**gridS.best_params_, random_state=0)

    elif regression_mod == 'KNN':
        min = 2
        max = 3
        if len(training_X) > 30:
            max = 16
            min = 5
        model = KNeighborsRegressor(n_neighbors=min)
        param = {'n_neighbors': np.arange(2, max, 2),
                 'weights': ('uniform', 'distance'),
                 'algorithm': ['auto'],  # 'ball_tree','kd_tree'),
                 'leaf_size': [10, 30, 50, 70, 90],
                 }
        if not test_mode:
            # print('Hyperparameter Tuning...')
            gridS = GridSearchCV(model, param)
            gridS.fit(training_X, training_Y)
            model = KNeighborsRegressor(**gridS.best_params_)

    elif regression_mod == 'SVR':
        model = SVR()
        param = {'kernel': ('linear', 'rbf'),
                 'degree': [2, 3, 4, 5],
                 'gamma': ('scale', 'auto'),
                 'coef0': [0, 2, 4, 6, 8, 10],
                 'epsilon': [0.01, 1]
                 }
        if not test_mode:
            # print('Hyperparameter Tuning...')
            gridS = GridSearchCV(model, param)
            gridS.fit(training_X, training_Y)
            model = SVR(**gridS.best_params_)

    elif regression_mod == 'LR':
        model = LinearRegression()
        param = {'fit_intercept': ('True', 'False'),
                 # 'normalize': ('True', 'False'),
                 'n_jobs': [1, -1]
                 }
        if not test_mode:
            # print('Hyperparameter Tuning...')
            gridS = GridSearchCV(model, param)
            gridS.fit(training_X, training_Y)
            model = LinearRegression(**gridS.best_params_)

    elif regression_mod == 'KR':
        x1 = np.arange(0.1, 5, 0.5)
        model = KernelRidge()
        param = {'alpha': x1,
                 'degree': [1, 2, 3, 4, 5, 6, 7, 8, 9],
                 'coef0': [1, 2, 3, 4, 5]
                 }
        if not test_mode:
            # print('Hyperparameter Tuning...')
            gridS = GridSearchCV(model, param)
            gridS.fit(training_X, training_Y)
            model = KernelRidge(**gridS.best_params_)
    return model


def init_dir(dir_name):
    """Creates directory if it does not exists"""
    if dir_name is not None:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

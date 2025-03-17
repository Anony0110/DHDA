from abc import ABC, abstractmethod
from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
# from river.neighbors import KNNRegressor
from skmultiflow.lazy import KNNRegressor
import numpy as np
from river.linear_model import LinearRegression
from river.tree import HoeffdingTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from utils.Meta_sparse_model_tf2 import MTLSparseModel
from utils.hyperparameter_tuning_MTSM import nn_l1_val
from utils.mlp_plain_model_tf2 import MLPPlainModel
from skmultiflow.meta import AdaptiveRandomForestRegressor
from river.ensemble import SRPRegressor as SRP
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd

pandas2ri.activate()

sk = importr('ScottKnottESD')


def get_attributes_result(data):
    X = data[:, :-1]
    Y = data[:, -1][:, np.newaxis]
    return X, Y


def hyper_tune(X_train, Y_train):
    # hyperparameter tuning for DeepPerf
    X_train1, X_train2, Y_train1, Y_train2 = train_test_split(X_train, Y_train, test_size=0.333)
    (N, n) = X_train.shape
    print('Tuning hyperparameters...')
    print('Step 1: Tuning the number of layers and the learning rate ...')
    config = dict()
    config['num_input'] = n
    config['num_neuron'] = 128
    config['lambda'] = 'NA'
    config['decay'] = 'NA'
    config['verbose'] = 0
    abs_error_all = np.zeros((15, 4))
    abs_error_all_train = np.zeros((15, 4))
    abs_error_layer_lr = np.zeros((15, 2))
    abs_err_layer_lr_min = 100
    count = 0
    layer_range = range(2, 15)
    # lr_range = [0.0025, 0.005, 0.0075, 0.01]
    lr_range = np.logspace(np.log10(0.0001), np.log10(0.1), 4)
    # lr_range = np.logspace(np.log10(0.0001), np.log10(0.1), 5)
    for n_layer in layer_range:
        config['num_layer'] = n_layer
        for lr_index, lr_initial in enumerate(lr_range):
            model = MLPPlainModel(config)
            model.build_train()
            model.train(X_train1, Y_train1, lr_initial)

            Y_pred_train = model.predict(X_train1)
            abs_error_train = np.mean(np.abs(Y_pred_train - Y_train1))
            abs_error_all_train[int(n_layer), lr_index] = abs_error_train

            Y_pred_val = model.predict(X_train2)
            abs_error = np.mean(np.abs(Y_pred_val - Y_train2))
            abs_error_all[int(n_layer), lr_index] = abs_error

        temp = abs_error_all_train[int(n_layer), :] / np.max(abs_error_all_train)
        temp_idx = np.where(abs(temp) < 0.0001)[0]
        if len(temp_idx) > 0:
            lr_best = lr_range[np.max(temp_idx)]
            err_val_best = abs_error_all[int(n_layer), np.max(temp_idx)]
        else:
            lr_best = lr_range[np.argmin(temp)]
            err_val_best = abs_error_all[int(n_layer), np.argmin(temp)]

        abs_error_layer_lr[int(n_layer), 0] = err_val_best
        abs_error_layer_lr[int(n_layer), 1] = lr_best

        if abs_err_layer_lr_min >= abs_error_all[int(n_layer), np.argmin(temp)]:
            abs_err_layer_lr_min = abs_error_all[int(n_layer),
            np.argmin(temp)]
            count = 0
        else:
            count += 1

        if count >= 2:
            break

    abs_error_layer_lr = abs_error_layer_lr[abs_error_layer_lr[:, 1] != 0]

    # Get the optimal number of layers
    n_layer_opt = layer_range[np.argmin(abs_error_layer_lr[:, 0])] + 3

    # Find the optimal learning rate of the specific layer
    config['num_layer'] = n_layer_opt
    for lr_index, lr_initial in enumerate(lr_range):
        model = MLPPlainModel(config)
        model.build_train()
        model.train(X_train1, Y_train1, lr_initial)

        Y_pred_train = model.predict(X_train1)
        abs_error_train = np.mean(np.abs(Y_pred_train - Y_train1))
        abs_error_all_train[int(n_layer), lr_index] = abs_error_train

        Y_pred_val = model.predict(X_train2)
        abs_error = np.mean(np.abs(Y_pred_val - Y_train2))
        abs_error_all[int(n_layer), lr_index] = abs_error

    temp = abs_error_all_train[int(n_layer), :] / np.max(abs_error_all_train)
    temp_idx = np.where(abs(temp) < 0.0001)[0]
    if len(temp_idx) > 0:
        lr_best = lr_range[np.max(temp_idx)]
    else:
        lr_best = lr_range[np.argmin(temp)]

    lr_opt = lr_best
    print('The optimal number of layers: {}'.format(n_layer_opt))
    print('The optimal learning rate: {:.4f}'.format(lr_opt))

    # Use grid search to find the right value of lambda
    lambda_range = np.logspace(-2, np.log10(1000), 30)
    error_min = np.zeros((1, len(lambda_range)))
    rel_error_min = np.zeros((1, len(lambda_range)))
    decay = 'NA'
    for idx, lambd in enumerate(lambda_range):
        val_abserror, val_relerror = nn_l1_val(X_train1, Y_train1,
                                               X_train2, Y_train2,
                                               n_layer_opt, lambd, lr_opt, max_epoch=2000)
        error_min[0, idx] = val_abserror
        rel_error_min[0, idx] = val_relerror

        # Find the value of lambda that minimize error_min
    lambda_f = lambda_range[np.argmin(error_min)]
    print('Step 2: Tuning the l1 regularized hyperparameter ...')
    print('The optimal l1 regularizer: {:.4f}'.format(lambda_f))

    # Solve the final NN with the chosen lambda_f on the training data
    config = dict()
    config['num_neuron'] = 128
    config['num_input'] = n
    config['num_layer'] = n_layer_opt
    config['lambda'] = lambda_f
    config['verbose'] = 1
    return config, lr_opt


def sequence_selection(history_data, n_runs=30):
    if len(history_data) == 1:
        return [0]
    else:
        scores = {}
        # Evaluate all available historical environment data to derive a training sequence
        for test_idx in range(len(history_data)):
            for target_idx in range(len(history_data)):
                if test_idx == target_idx:
                    continue
                for i in range(n_runs):
                    # print(f"target env{target_idx} model env{test_idx} run turns{i + 1}")
                    test_env = history_data[test_idx]
                    x_train_pool = test_env[:, :-1]
                    y_train_pool = test_env[:, -1][:, np.newaxis]

                    target_env = history_data[target_idx]
                    x_test_pool = target_env[:, :-1]
                    y_test_pool = target_env[:, -1][:, np.newaxis]

                    X_train, _, y_train, _ = train_test_split(x_train_pool, y_train_pool,
                                                              train_size=0.7,
                                                              shuffle=True)

                    X_test, _, y_test, _ = train_test_split(x_test_pool, y_test_pool,
                                                            test_size=0.3,
                                                            shuffle=True)
                    reg = DecisionTreeRegressor(random_state=42)
                    reg.fit(X_train, y_train)

                    y_pred = reg.predict(X_test)
                    score = mean_absolute_percentage_error(y_test, y_pred)
                    if test_idx in scores:
                        scores[test_idx].append(score)
                    else:
                        scores[test_idx] = [score]
        ranking = None
        # print(scores)

        # use Scott-Knott test to rank each single-environment model to decide the best sequence
        if len(history_data) >= 3: # Scott-Knott test only works for more than 2 groups
            result = pd.DataFrame(scores)
            r_sk = sk.sk_esd(result)
            column_order = list(r_sk[3])
            ranking = pd.DataFrame(
                {
                    "technique": [result.columns[i - 1] for i in column_order],
                    "rank": r_sk[1].astype("int"),
                }
            )
            # print(ranking)
            result = ranking.values
            sequence = result[:, 0].tolist()
        else:
            sequence = sorted(scores, key=lambda x: scores[x], reverse=True)
        # print(sequence)
        return sequence


# In the DHDA framework architecture, regression models supporting incremental updates are required to subclass the BaseModel(ABC) abstract class
class BaseModel(ABC):
    @abstractmethod
    def update(self, x, y):
        pass

    @abstractmethod
    def fit(self, x, y):
        pass

    @abstractmethod
    def predict(self, x):
        pass


class LinearRegressor(BaseModel):
    def __init__(self):
        self.n_feature = None
        self.model = LinearRegression(intercept_lr=.1)
        self.max_y = None

    def update(self, x, y):
        if len(y) == 1:
            data_structure = {}
            for index, data in enumerate(x):
                for i, feature in enumerate(data):
                    data_structure[i] = feature
                self.model.learn_one(data_structure, y[index])
        else:
            y = np.divide(y, self.max_y)
            n_count, n_feature = x.shape
            self.n_feature = n_feature
            columns = list(range(self.n_feature))
            df = pd.DataFrame(x, columns=columns)
            y_s = np.squeeze(y)
            s = pd.Series(y_s, name='output')
            self.model.learn_many(df, s)

    def fit(self, x, y):
        self.max_y = max(y)
        self.model = LinearRegression(intercept_lr=.1)
        self.update(x, y)

    def predict(self, x):
        data_structure = {}
        for index, data in enumerate(x):
            for i, feature in enumerate(data):
                data_structure[i] = feature
        return [self.model.predict_one(x=data_structure) * self.max_y]


class KNNRegression(BaseModel):
    def __init__(self):
        self.n_feature = None
        self.model = KNNRegressor()

    def update(self, x, y):
        self.model.partial_fit(x, y)

    def fit(self, x, y):
        self.model = KNNRegressor()
        self.model.fit(x, y)

    def predict(self, x):
        pre_y = self.model.predict(x)
        return pre_y


class HoeffdingTree(BaseModel):
    def __init__(self):
        self.n_feature = None
        self.model = HoeffdingTreeRegressor(grace_period=100)

    def update(self, x, y):
        n_count, self.n_feature = x.shape
        data_structure = {}
        for index, data in enumerate(x):
            for i, feature in enumerate(data):
                data_structure[i] = feature
            self.model.learn_one(data_structure, y[index])

    def fit(self, x, y):
        self.model = HoeffdingTreeRegressor(grace_period=100)
        self.update(x, y)

    def predict(self, x):
        data_structure = {}
        for index, data in enumerate(x):
            for i, feature in enumerate(data):
                data_structure[i] = feature
        pre_y = self.model.predict_one(x=data_structure)
        return [pre_y]


class RandomForest(BaseModel):
    def __init__(self, model_reuse=True):
        self.n_feature = None
        self.model_reuse = model_reuse
        self.model = RandomForestRegressor(warm_start=True, random_state=3)

    def update(self, x, y):
        self.model.n_estimators += 5
        self.model.fit(x, y)

    def fit(self, x, y):
        self.model = RandomForestRegressor(warm_start=True, random_state=3)
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)


class DeepPerf(BaseModel):
    def __init__(self, test_model=True):
        self.test_model = test_model
        self.n_feature = None
        self.n_count = None
        self.model = None
        self.config = None
        self.lr = None

    def update(self, x, y):
        weights, bias = self.model.get_weights()
        model = MTLSparseModel(self.config)
        model.read_weights(weights, bias)
        model.train(x, y, self.lr, max_epoch=1000)
        self.model = model

    def fit(self, x, y):
        self.n_count, self.n_feature = x.shape
        if not self.test_model:
            config, lr = hyper_tune(x, y)
        else:
            config = dict()
            config['num_neuron'] = 128
            config['num_input'] = self.n_feature
            config['num_layer'] = 8
            config['lambda'] = 0.123
            config['verbose'] = 1
            lr = 0.01
        self.model = MTLSparseModel(config)
        self.config = config
        self.lr = lr
        self.model.build_train()
        self.model.train(x, y, self.lr, max_epoch=2000)

    def predict(self, x):
        return self.model.predict(x)


def build_incremental_model(regressor='RF', model_reuse=True):
    model = None
    if regressor == 'RF':
        return RandomForest(model_reuse)
    if regressor == 'LR':
        return LinearRegressor()
    if regressor == 'HT':
        model = HoeffdingTree()
    if regressor == 'KNN':
        model = KNNRegression()
    if regressor == 'DeepPerf':
        model = DeepPerf()
    return model

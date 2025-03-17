import copy
import math

from scipy import stats
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from imblearn.over_sampling import SMOTE
from sklearn.metrics import mean_absolute_percentage_error
import configs
from utils.general import recursive_dividing
from skmultiflow import drift_detection
from base_model.basemodels import build_incremental_model


def get_whole_data(path):
    df = pd.read_csv(path)
    ndarray = df.values
    return ndarray


def get_attributes_result(data):
    X = data[:, :-1]
    Y = data[:, -1][:, np.newaxis]
    return X, Y


def combine_x_y(x, y):
    return np.hstack((x, y))


def combine_data(old, new):
    return np.vstack((old, new))


def split_data(data, rate):
    pass


def smo_data(x, ylabel):
    smo = SMOTE(random_state=1, k_neighbors=3)
    return smo.fit_resample(x, ylabel)


def get_tree_info(tree, d):
    node_count = 2 ** d - 1
    if d == 2:
        left_node = tree.left_child[0]
        right_node = tree.right_child[0]
        left_feature = tree.left_feature[left_node]
        right_feature = tree.right_feature[right_node]
        feature = [tree.feature[0], tree.feature[left_feature], tree.feature[right_feature]]
        threshold = [tree.threshold[0], tree.threshold[left_feature], tree.threshold[right_feature]]
    else:
        # warning: unsupported d != 1 or 2
        feature = [tree.feature[0]]
        threshold = [tree.threshold[0]]
    return feature, threshold


def info_compare(old, new):
    result = [old[0].__eq__(new[0]), old[1].__eq__(new[1])]
    return result  # todo:bug


def clusters_compare(old, new):
    similarity_list = []
    for index, cluster in enumerate(new):
        new_set = set(cluster)
        old_set = set(old[index])
        same_count = len(new_set.intersection(old_set))
        similarity_list.append(same_count / len(new_set))
    return similarity_list


def cart_change_test(G_best, G_second, n, DELTA):
    hoeffding_bound = math.sqrt((math.log(1.0 / DELTA)) / (2.0 * n))
    return G_best - G_second > hoeffding_bound


def my_cart_change_test(G_best, G_second, n1, n2, DELTA):
    my_hoeffding_bound = math.sqrt(math.log(1.0 / DELTA) / (4.0 / (1.0 / n1 + 1.0 / n2)))
    # hoeffding_bound = math.sqrt((math.log(1.0 / DELTA)) / (4.0 * (2.0 / (1.0 / n1 + 1.0 / n2))))
    return G_best - G_second > my_hoeffding_bound


def compute_hoeffding_bound(G_best, G_second, range_val, confidence, n):
    hoeffding_bound = np.sqrt((range_val * range_val * np.log(1.0 / confidence)) / (2.0 * n))
    return G_best - G_second > hoeffding_bound


baseline_models = ['LR', 'HT', 'KNN', 'RF']
class DHDA_Regressor:
    def __init__(self, depth=1, min_clock=32, base_model="RF", Hybrid_Frequency=3,
                 Hybrid_Update=True, Upper_Adapt=True, Lower_Adapt=True):
        self.base_model = base_model
        self.test_mode = True
        self.MODEL_REUSE = True
        self.Upper_Adapt = Upper_Adapt
        self.Lower_Adapt = Lower_Adapt
        self.HYBRID_UPDATE = Hybrid_Update
        self.init = True
        self.models = list()
        self.CART = None  # to extract the dividing conditions
        self.rfc = None  # to classify streaming samples
        self.clusters_index = list()
        # self.clusters = list()
        self.depth = depth
        self.clusters_num = 0
        self.feature_count = 0
        self.current_data_index = None
        self.whole_data = None
        self.clock = 0
        self.min_clock = min_clock
        self.detected_change_point_dict = dict()
        self.detected_cart_change_point_list = list()
        self.learn_threshold = configs.LEARNING_THRESHOLD
        self.recent_error = list()
        self.mean_error = list()
        self.count_test = list()
        self.val_list = list()
        self.warning_detectors = []
        self.drift_detectors = []
        self.modes = []
        self.data_size = []
        self.model_manager = ModelsManager()
        self.retrain_clock = Hybrid_Frequency * min_clock

    def divide_cluster(self, x, y):
        # current data -> divisions
        DT = DecisionTreeRegressor(random_state=3, criterion='squared_error', splitter='best')
        DT.fit(x, y)
        self.CART = DT
        tree_ = DT.tree_
        cluster_indexes_all = []  # for recursive algorithms
        cluster_indexes_all = recursive_dividing(0, 1, tree_, x, range(len(x)), self.depth,
                                                 0, cluster_indexes_all)
        return cluster_indexes_all

    def fit_basemodels(self, clustersx, clustersy):
        # train local model for each division
        self.models = list()
        for i in range(len(self.clusters_index)):
            # print('Training DNN for division {}... ({} samples)'.format(i + 1, len(clustersx[i])))
            model = build_incremental_model(self.base_model, model_reuse=self.MODEL_REUSE)
            model.fit(clustersx[i], clustersy[i])
            self.models.append(model)

    def init_detectors(self):
        # warning - drift detector for each division
        for i in range(len(self.models)):
            """detector = drift_detection.ADWIN(delta=configs.DELTA)
            self.detectors.append(detector)"""
            warning_detector = drift_detection.ADWIN(0.01)
            warning_detector.set_clock(16)
            drift_detector = drift_detection.ADWIN(0.005)
            self.warning_detectors.append(warning_detector)
            self.drift_detectors.append(drift_detector)
            self.modes.append(False)
            self.data_size.append(0)

    def fit_rfc(self, x, y):
        rfc = RandomForestClassifier(criterion='gini', random_state=3)
        self.rfc = rfc
        self.rfc.fit(x, y)
        return

    def fit_model(self, x, y):
        # training DHDA
        self.whole_data = combine_x_y(x, y)
        self.current_data_index = list(range(len(self.whole_data)))
        self.clock = len(y)
        x = np.array(x)
        (self.data_count, self.feature_count) = x.shape
        y = np.array(y)

        # dividing
        self.clusters_index = self.divide_cluster(x, y)
        y_label = list()
        for i in range(len(x)):
            for label, cluster in enumerate(self.clusters_index):
                if i in cluster:
                    y_label.append(label)
                    continue

        if len(self.clusters_index) == 2:
            if len(self.clusters_index[0]) > 5 and len(self.clusters_index[1]) > 5:
                rfc_x, rfc_y = smo_data(x, y_label)
            else:
                rfc_x = x
                rfc_y = y_label
        else:
            rfc_x = x
            rfc_y = y_label
        self.fit_rfc(rfc_x, rfc_y)
        train_x = list()
        train_y = list()
        for c in self.clusters_index:
            train_data_temp = self.whole_data[c]
            x_temp, y_temp = get_attributes_result(train_data_temp)
            train_x.append(x_temp)
            train_y.append(y_temp)
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        self.val_list = [0] * len(y)
        # training local models
        if self.base_model in baseline_models:
            self.fit_basemodels(train_x, train_y)
        else:
            print("err base model")
            exit()
        self.init_detectors()
        self.model_manager.change_to_cart(get_tree_info(self.CART.tree_, 1))
        for i in range(len(self.clusters_index)):
            self.recent_error.append(list())
            self.mean_error.append(0)
            self.count_test.append(0)

    def predict(self, x):
        # get configurations' label and predict
        test_labels = self.rfc.predict(x)
        pre_y = []
        for i, label in enumerate(test_labels):
            # print(f"pre in cluster{test_labels}")
            y = self.models[label].predict(x[i][np.newaxis, :])
            pre_y.append(y[0])
        return pre_y

    def learn_data(self, x, y, val):
        # learn when new data are available
        # val: absolute percentage error
        new_data = combine_x_y(x, y)
        self.whole_data = combine_data(self.whole_data, new_data)
        self.val_list.append(val)

        for i, data in enumerate(x):
            self.clock += 1
            self.current_data_index.append(len(self.whole_data) - 1)
            cluster_index = self.rfc.predict(data[np.newaxis, :])[0]
            # self.clusters_index[cluster_index].append(len(self.whole_data) - 1)
            self.recent_error[cluster_index].append(val)
            n_count = self.count_test[cluster_index]
            self.count_test[cluster_index] += 1
            self.mean_error[cluster_index] = (self.mean_error[cluster_index] * n_count + val) / (n_count + 1)
            if len(self.recent_error[cluster_index]) > 32:
                self.recent_error[cluster_index].pop(0)
            sq_val = val
            if self.Lower_Adapt:
                weights = stats.poisson.rvs(mu=2 ** self.depth, size=1)[0]
                if self.modes[cluster_index]:
                    self.data_size[cluster_index] += 1
                for times in range(weights):
                    self.warning_detectors[cluster_index].add_element(sq_val)
                    if self.warning_detectors[cluster_index].detected_change():
                        # print(f"warning happen in index{cluster_index} at index{self.clock} at{self.clock / 32}")
                        self.modes[cluster_index] = True
                        self.data_size[cluster_index] = 0
                        self.warning_detectors[cluster_index].reset()

                    self.drift_detectors[cluster_index].add_element(sq_val)

                    if self.drift_detectors[cluster_index].detected_change():
                        # Drift detected in the cluster_indexth division
                        mean_recent = np.mean(self.recent_error[cluster_index])
                        mean_old = self.mean_error[cluster_index]
                        if mean_recent < mean_old:  # downward drift
                            self.modes[cluster_index] = False
                            self.drift_detectors[cluster_index].reset()
                            self.warning_detectors[cluster_index].reset()
                            for val in self.recent_error[cluster_index]:
                                self.warning_detectors[cluster_index].add_element(val)
                                self.drift_detectors[cluster_index].add_element(val)
                                self.warning_detectors[cluster_index].add_element(val)
                                self.drift_detectors[cluster_index].add_element(val)
                            print(
                                f"downward drift happen in index{cluster_index} at index{self.clock} at{self.clock / 32}")
                        else:
                            print(f"drift happen in index{cluster_index} at index{self.clock} at{self.clock / 32}")
                            self.modes[cluster_index] = False
                            self.drift_detectors[cluster_index].reset()
                            self.warning_detectors[cluster_index].reset()

                            if cluster_index not in self.detected_change_point_dict:  # record the change point detection result
                                self.detected_change_point_dict[cluster_index] = list()
                            self.detected_change_point_dict[cluster_index].append(self.clock)

                            whole_size = len(self.clusters_index[cluster_index])
                            current_window_size = self.data_size[cluster_index]
                            self.data_size[cluster_index] = 0

                            if current_window_size < configs.MIN_CLUSTERS_SAMPLES:
                                current_window_size = configs.MIN_CLUSTERS_SAMPLES

                            if current_window_size > configs.MAX_CLUSTERS_SAMPLES:
                                current_window_size = configs.MAX_CLUSTERS_SAMPLES

                            delete_data_size = whole_size - current_window_size
                            delete_data = self.clusters_index[cluster_index][:delete_data_size]
                            self.val_list = self.val_list[delete_data_size:]
                            self.mean_error[cluster_index] = 0
                            self.count_test[cluster_index] = 0
                            remain_data = self.clusters_index[cluster_index][-current_window_size:]
                            # update cluster index
                            self.clusters_index[cluster_index] = remain_data
                            old_model = build_incremental_model(self.base_model)
                            old_x = self.whole_data[
                                        delete_data[int(delete_data_size * 1 / 4):int(delete_data_size * 3 / 4)]][:,
                                    :-1]
                            old_y = self.whole_data[
                                        delete_data[int(delete_data_size * 1 / 4):int(delete_data_size * 3 / 4)]][:, -1]
                            old_model.fit(old_x, old_y)
                            for item in delete_data:
                                while item in self.current_data_index:
                                    self.current_data_index.remove(item)
                            train_x, train_y = get_attributes_result(
                                self.whole_data[self.clusters_index[cluster_index]])
                            # retrain local model
                            # dnn_x, dnn_y = self.single_DNN_data_preprocessing(new_x, new_y)
                            if self.MODEL_REUSE:
                                model = self.model_manager.get_model_by_accuracy(train_x, train_y, cluster_index)
                                self.model_manager.add_model_in_current_cart(old_model, cluster_index)
                                if model is not None:
                                    print("reuse successfully")
                                    if self.base_model in baseline_models:
                                        model.update(train_x, train_y)
                                        self.models[cluster_index] = model
                                    else:
                                        print("base model err 2")
                                        exit()

                                else:
                                    if self.base_model in baseline_models:
                                        self.models[cluster_index].fit(train_x, train_y)
                                    else:
                                        print("base model err 2")
                                        exit()
                            else:  # when model reuse is not allowed:
                                if self.base_model in baseline_models:
                                    self.models[cluster_index].fit(train_x, train_y)
                                else:
                                    print("base model err 2")
                                    exit()
                            print(f"warning window size: {len(train_y)}")
            else:  # if lower adaption is not allowed, use most resent 256 sample to train local model
                self.current_data_index = self.current_data_index[-256:]

            if self.clock % self.min_clock == 0:
                # updated in batches
                current_x, current_y = get_attributes_result(self.whole_data[self.current_data_index])
                current_DT = DecisionTreeRegressor(random_state=3, criterion='squared_error', splitter='best')
                current_DT.fit(current_x, current_y)
                current_tree_info = get_tree_info(current_DT.tree_, 1)
                old_tree_info = get_tree_info(self.CART.tree_, 1)

                if np.mean(self.recent_error[cluster_index]) >= self.learn_threshold or len(
                        self.clusters_index[cluster_index]) < 128:
                    if self.HYBRID_UPDATE:
                        clusters_index = list()
                        clusters_index = recursive_dividing(0, 1, self.CART.tree_, current_x, self.current_data_index,
                                                            max_depth=self.depth, min_samples=0,
                                                            cluster_indexes_all=clusters_index)
                        self.clusters_index = clusters_index
                    y_label = list()
                    if self.Upper_Adapt:
                        for index in self.current_data_index:
                            for label, cluster in enumerate(self.clusters_index):
                                if index in cluster:
                                    y_label.append(label)
                                    continue
                        self.fit_rfc(current_x, y_label)

                    train_x = list()
                    train_y = list()
                    for c in self.clusters_index:
                        train_data_temp = self.whole_data[c]
                        x_temp, y_temp = get_attributes_result(train_data_temp)
                        train_x.append(x_temp)
                        train_y.append(y_temp)

                    if self.HYBRID_UPDATE:  # lower update
                        if self.clock % self.retrain_clock == 0:
                            self.models = list()
                            for mi in range(len(self.clusters_index)):
                                model = build_incremental_model(regressor=self.base_model,
                                                                model_reuse=self.MODEL_REUSE)
                                model.fit(train_x[mi], train_y[mi])
                                self.models.append(model)
                        else:
                            for index, xs in enumerate(train_x):
                                new_x = xs[-self.min_clock:]
                                new_y = train_y[index][-self.min_clock:]
                                self.models[index].update(new_x, new_y)

                cart_result = info_compare(old_tree_info, current_tree_info)
                if not (cart_result[0] and cart_result[1]):
                    # Upper change
                    isChange = True
                    old_best_feature = self.CART.tree_.feature[0]
                    new_best_feature = current_DT.tree_.feature[0]
                    G_best = current_DT.feature_importances_[new_best_feature]
                    G_second = current_DT.feature_importances_[old_best_feature]
                    if len(self.clusters_index) == 2:
                        if (len(self.clusters_index[0]) / len(self.clusters_index[1]) > 3 or
                                len(self.clusters_index[1]) / len(self.clusters_index[0]) > 3):
                            isChange = False

                        if not my_cart_change_test(G_best, G_second, len(self.clusters_index[0]),
                                                   len(self.clusters_index[1]), 0.0000001):
                            isChange = False
                    if isChange and self.Upper_Adapt:
                        print(
                            f"------------------------------CART change detected in {self.clock}------------------------------")
                        print(f"the old CART feature: {old_best_feature}, with weight{G_second}")
                        print(f"the new CART feature: {new_best_feature}, with weight{G_best}")
                        print("the old clusters info:")
                        for index in range(len(self.clusters_index)):
                            print(
                                f"the {index}-th cluster contain samples count {len(self.clusters_index[index])}, from {self.clusters_index[index][0]} to {self.clusters_index[index][-1]}")
                        self.detected_cart_change_point_list.append(self.clock)
                        self.CART = current_DT
                        clusters_index = list()
                        clusters_index = recursive_dividing(0, 1, current_DT.tree_, current_x, self.current_data_index,
                                                            max_depth=self.depth, min_samples=0,
                                                            cluster_indexes_all=clusters_index)

                        self.clusters_index = clusters_index
                        print("the new clusters info:")
                        for index in range(len(self.clusters_index)):
                            print(
                                f"the {index}-th cluster contain samples count {len(self.clusters_index[index])}, from {self.clusters_index[index][0]} to {self.clusters_index[index][-1]}")
                        y_label = list()
                        for index in self.current_data_index:
                            for label, cluster in enumerate(self.clusters_index):
                                if index in cluster:
                                    y_label.append(label)
                                    continue
                        self.fit_rfc(current_x, y_label)

                        train_x = list()
                        train_y = list()
                        self.model_manager.change_to_cart(current_tree_info)
                        # get new clusters
                        for c in self.clusters_index:
                            train_data_temp = self.whole_data[c]
                            x_temp, y_temp = get_attributes_result(train_data_temp)
                            train_x.append(x_temp)
                            train_y.append(y_temp)
                        if self.MODEL_REUSE:
                            for m in range(len(self.models)):
                                model = self.model_manager.get_model_by_accuracy(train_x[m], train_y[m], m)
                                if model is None:  #or len(train_y[m] > 64)
                                    if self.base_model in baseline_models:
                                        self.models[m].fit(train_x[m], train_y[m])
                                else:
                                    if self.base_model in baseline_models:
                                        self.models[m].update(train_x[m], train_y[m])
                                    else:
                                        print("err3")
                                        exit()
                        else:
                            if self.base_model in baseline_models:
                                self.fit_basemodels(train_x, train_y)
                            else:
                                print("err3")
                                exit()


class ModelsManager:
    def __init__(self):
        self.current_cart_index = 0
        self.carts_list = []
        self.models = []
        self.similar_threshold = 0.02

    def cart_info_exit(self, cart_info):
        for cart in self.carts_list:
            result = info_compare(cart, cart_info)
            if result[0] and result[1]:
                return True
        return False

    def get_cart_index(self, cart_info):
        for index, cart in enumerate(self.carts_list):
            result = info_compare(cart, cart_info)
            if result[0] and result[1]:
                return index
        return None

    def change_to_cart(self, cart_info):
        if not self.cart_info_exit(cart_info):
            self.carts_list.append(cart_info)
            self.models.append([[], []])
            self.current_cart_index = len(self.carts_list) - 1
        else:
            self.current_cart_index = self.get_cart_index(cart_info)

    def add_model_in_current_cart(self, model, index):
        self.models[self.current_cart_index][index].append(model)


    def get_model_by_accuracy(self, x, y, cluster_index):
        if len(self.models[self.current_cart_index][cluster_index]) == 0:
            return None
        err_list = []
        y = np.array(y)
        for model in self.models[self.current_cart_index][cluster_index]:
            pre_y = []
            for i, config in enumerate(x):
                y_temp = model.predict(x[i][np.newaxis, :])
                pre_y.append(y_temp[0])
            pre_y = np.array(pre_y)
            err = mean_absolute_percentage_error(y_true=y, y_pred=pre_y)
            err_list.append(err)
        min_err = 100000
        min_index = -1
        for index, err in enumerate(err_list):
            if err < min_err:
                min_err = err
                min_index = index
        # self.models[self.current_cart_index][index].append(self.models[self.current_cart_index][index][max_index])
        if min_err > 0.02:
            return None
        return copy.deepcopy(self.models[self.current_cart_index][cluster_index][min_index])

    def get_model_by_accuracy_threshold(self, x, y, cluster_index):
        if len(self.models[self.current_cart_index][cluster_index]) == 0:
            return None
        err_list = []
        y = np.array(y)
        for model in self.models[self.current_cart_index][cluster_index]:
            pre_y = []
            for i, config in enumerate(x):
                y = model.predict(x[i][np.newaxis, :])
                pre_y.append(y[0])
            pre_y = np.array(pre_y)
            err = np.sum(abs(y - pre_y))
            err_list.append(err)
        min_err = 0
        min_index = -1
        for index, err in enumerate(err_list):
            if err < min_err:
                min_err = err
                min_index = index
        # self.models[self.current_cart_index][index].append(self.models[self.current_cart_index][index][max_index])
        if min_err < self.similar_threshold:
            return copy.deepcopy(self.models[self.current_cart_index][cluster_index][min_index])
        else:
            return None


class HistoryModel:
    def __init__(self):
        self.model = None
        self.mean = 0
        self.val = 0

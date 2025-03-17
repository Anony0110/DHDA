import math
import random
import time
import numpy as np
import pandas
import pandas as pd
import sklearn.metrics
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from skmultiflow import drift_detection
import DHDA
from skmultiflow.meta import AdaptiveRandomForestRegressor
from river.forest import ARFRegressor
from utils.general import process_training_data


dataset_seed = 108
sample_seed = 17
"""random.seed(dataset_seed)  # for choose datasets
np.random.seed(sample_seed)"""  # for sampling in a dataset
"""
high-density cart change seed:49
dataset: x264_15.csv
"""
env_list1 = ["drift_data\\data1\\sac_2.csv",
             "drift_data\\data1\\sac_4.csv",
             "drift_data\\data1\\sac_5.csv",
             "drift_data\\data1\\sac_6.csv",
             "drift_data\\data1\\sac_7.csv",
             "drift_data\\data1\\sac_8.csv",
             "drift_data\\data1\\sac_9.csv"]

env_list2 = ["drift_data\\data2\\x264_0.csv",
             "drift_data\\data2\\x264_1.csv",
             "drift_data\\data2\\x264_2.csv",
             "drift_data\\data2\\x264_3.csv",
             "drift_data\\data2\\x264_4.csv",
             "drift_data\\data2\\x264_5.csv",
             "drift_data\\data2\\x264_6.csv",
             "drift_data\\data2\\x264_7.csv",
             "drift_data\\data2\\x264_8.csv",
             "drift_data\\data2\\x264_9.csv",
             "drift_data\\data2\\x264_10.csv",
             "drift_data\\data2\\x264_11.csv",
             "drift_data\\data2\\x264_12.csv",
             "drift_data\\data2\\x264_13.csv",
             "drift_data\\data2\\x264_14.csv",
             "drift_data\\data2\\x264_15.csv",
             "drift_data\\data2\\x264_16.csv",
             "drift_data\\data2\\x264_17.csv",
             "drift_data\\data2\\x264_18.csv",
             "drift_data\\data2\\x264_19.csv",
             "drift_data\\data2\\x264_20.csv", ]

env_list3 = ["drift_data\\data3\\storm-obj1_feature6.csv",
             "drift_data\\data3\\storm-obj1_feature7.csv",
             "drift_data\\data3\\storm-obj1_feature8.csv",
             "drift_data\\data3\\storm-obj2_feature9.csv",
             "drift_data\\data3\\storm-obj2_feature6.csv",
             "drift_data\\data3\\storm-obj1_feature7.csv",
             "drift_data\\data3\\storm-obj2_feature8.csv",
             "drift_data\\data3\\storm-obj2_feature9.csv"]

env_list4 = ["drift_data\\data4\\spear_0.csv",
             "drift_data\\data4\\spear_1.csv",
             "drift_data\\data4\\spear_2.csv",
             "drift_data\\data4\\spear_3.csv",
             "drift_data\\data4\\spear_4.csv",
             "drift_data\\data4\\spear_5.csv",
             "drift_data\\data4\\spear_6.csv",
             "drift_data\\data4\\spear_7.csv",
             "drift_data\\data4\\spear_8.csv",
             "drift_data\\data4\\spear_9.csv"]

env_list5 = ["drift_data\\data5\\sqlite_10.csv",
             "drift_data\\data5\\sqlite_11.csv",
             "drift_data\\data5\\sqlite_16.csv",
             "drift_data\\data5\\sqlite_17.csv",
             "drift_data\\data5\\sqlite_18.csv",
             "drift_data\\data5\\sqlite_19.csv",
             "drift_data\\data5\\sqlite_44.csv",
             "drift_data\\data5\\sqlite_45.csv",
             "drift_data\\data5\\sqlite_52.csv",
             "drift_data\\data5\\sqlite_59.csv",
             "drift_data\\data5\\sqlite_73.csv",
             "drift_data\\data5\\sqlite_79.csv",
             "drift_data\\data5\\sqlite_94.csv",
             "drift_data\\data5\\sqlite_96.csv",
             "drift_data\\data5\\sqlite_97.csv"]

env_list6 = ["drift_data\\data6\\nginx1.csv",
             "drift_data\\data6\\nginx2.csv",
             "drift_data\\data6\\nginx3.csv",
             "drift_data\\data6\\nginx4.csv"]

env_list7 = ["drift_data\\data7\\exastencils1.csv",
             "drift_data\\data7\\exastencils2.csv",
             "drift_data\\data7\\exastencils3.csv",
             "drift_data\\data7\\exastencils4.csv"]

env_list8 = ["drift_data\\data8\\deeparch1.csv",
             "drift_data\\data8\\deeparch2.csv",
             "drift_data\\data8\\deeparch3.csv"
             ]


if __name__ == "__main__":
    save_file = False  # to save the results, set to True
    save_path = "result\\DHDA.csv"
    env_list = env_list2  # Select testing system
    n_environment = 5  # Randomly select n environments from the system dataset list
    regressor = "RF"  # Local model
    n_drifts = n_environment - 1  # Actual number of drifts
    init_train_count = 50  # Number of initialization samples
    max_count = 4000
    max_sampling_ratio = 0.9
    min_sampling_ratio = 0.6
    run_time = 30
    result = {"mMAPE":[], "Time_cost":[]}
    for t in range(run_time):
        random.seed(t * 2 + 1)
        np.random.seed(t * 2 + 1)
        raw = process_training_data(env_list, n_drifts=n_drifts, init_train_count=init_train_count,
                                    min_occupation=min_sampling_ratio, max_occupation=max_sampling_ratio,
                                    max_count=max_count, regressor=regressor)
        y = list()
        dataset_name = env_list[0].split('\\')[-1]
        print("the datasets are...")
        x_train = raw['x_train']
        y_train = raw['y_train']
        x_test = raw['x_test']
        y_test = raw['y_test']
        count_list = raw['count_list']
        print(count_list)
        test_size = len(y_test)
        print("DHDA")
        err_list = []
        time1 = time.time()
        dhda = DHDA.DHDA_Regressor(depth=1, min_clock=32, base_model=regressor, Hybrid_Frequency=3)
        dhda.fit_model(x_train, y_train)  # train DHDA
        c = 0
        for i in range(test_size):
            c += 1
            y_pre = dhda.predict(x_test[i][np.newaxis, :])
            if y_test[i] != 0:
                val = ((abs(y_pre - y_test[i])[0]) / y_test[i])[0]
                err_list.append(val)
                dhda.learn_data(x_test[i][np.newaxis, :], y_test[i][np.newaxis, :], val)
            if c % 32 == 0:
                err = np.mean(err_list)
                y.append(err)
                err_list = list()

        time2 = time.time()
        time_cost = time2 - time1
        offset = int(init_train_count / 32) + 2
        x = list(range(offset, offset + len(y)))
        plt.plot(x, y, label='DHDA')
        plt.title(f'{dataset_name} {regressor}')
        plt.ylabel('MAPE')
        DHDA_mMAPE = np.mean(y)
        performance_list = [round(DHDA_mMAPE, 4)]
        plt.xlabel("Timesteps")
        plt.legend()
        plt.show()
        print(f"run time {t}: mMape = {DHDA_mMAPE}, Time cost = {time_cost}")
        result["mMAPE"].append(DHDA_mMAPE)
        result["Time_cost"].append(time_cost)
    if save_file:
        result = pd.DataFrame(result)
        result.to_csv(save_path)

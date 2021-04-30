import data_processing_utils as ds_utils
import arima_utils as ar_utils
import time as times
import psutil

file = 'anomalous_detection_NASA/resources/Averaged_BearingTest_Dataset.csv'
header_set = ['datetime', 'Bearing1']
drop_header_set = []
check_ds = False


def task1():
    print('loading dataset ...')
    ds = ds_utils.load_dataset(file, header_set, drop_header_set, 'datetime', 983)

    print(ds.head())

    print('splitting dataset ...')
    train_set, test_set = ds_utils.prepare_dataset(ds, 0.2, False, 'Bearing1')

    print("train_set", train_set.shape)
    print("test_set", test_set.shape)

    return train_set, test_set


def task3(train_set, test_set, algo):
    ar_utils.test_dataset(train_set['Bearing1'])

    ar_utils.plot_series(train_set, "train set")
    ar_utils.plot_series(test_set, "test set")

    if check_ds == True:
        print('adf fuller testing ...')
        ar_utils.adfuller_test(train_set['Bearing1'])
        ar_utils.plot_pacf(train_set['Bearing1'])
        ar_utils.plot_acf(train_set['Bearing1'])

    t0 = times.time()
    cpu_t0 = psutil.cpu_percent(interval=0.5)
    model = ar_utils.generate_model(train_set, 0, 3, 3, 'Bearing1', algo)
    t1 = times.time()
    cpu_t1 = psutil.cpu_percent(interval=0.5)
    train_t = t1 - t0
    cpu_train_perc = abs(cpu_t1 - cpu_t0)

    cpu_p0 = psutil.cpu_percent(interval=0.5)
    pred_set = ar_utils.predict(model, len(test_set))
    t2 = times.time()
    cpu_p1 = psutil.cpu_percent(interval=0.5)
    pred_t = t2 - t1
    cpu_pred_perc = abs(cpu_p1 - cpu_p0)

    ar_utils.plot_difference_model(test_set, pred_set, algo)

    return train_t, pred_t, cpu_train_perc, cpu_pred_perc


train_set, test_set = task1()
train_mle_t, pred_mle_t, cpu_train_mle_perc, cpu_pred_mle_perc = task3(train_set, test_set, 'MLE')

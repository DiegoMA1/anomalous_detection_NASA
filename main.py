import data_processing_utils as ds_utils
import arima_utils as ar_utils
import time as times
import psutil

file = 'resources/http.log'
header_set = ['ts', 'uid', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p',
              'trans_depth', 'method', 'host', 'uri', 'referrer', 'user_agent',
              'request_body_len', 'response_body_len', 'status_code', 'status_msg',
              'info_code', 'info_msg', 'filename', 'tags', 'username',
              'password', 'proxied', 'orig_fuids', 'orig_mime_types', 'resp_fuids',
              'resp_mime_types', 'sample']

drop_header_set = ['uid', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p',
                   'trans_depth', 'method', 'host', 'uri', 'referrer', 'user_agent',
                   'status_code', 'status_msg',
                   'info_code', 'info_msg', 'filename', 'tags', 'username',
                   'password', 'proxied', 'orig_fuids', 'orig_mime_types', 'resp_fuids',
                   'resp_mime_types', 'sample']

check_ds = False


def task1():
    print('loading dataset ...')
    ds = ds_utils.load_dataset(file, header_set, drop_header_set, 'ts', 4023)

    print('spliting dataset ...')
    train_set, test_set = ds_utils.prepare_dataset(ds, 0.2, False, 'response_body_len')

    return train_set, test_set


def task3(train_set, test_set, algo):
    ar_utils.test_dataset(train_set['response_body_len'])

    ar_utils.plot_series(train_set, "train set")
    ar_utils.plot_series(test_set, "test set")

    if check_ds == True:
        print('adf fuller testing ...')
        ar_utils.adfuller_test(train_set['response_body_len'])
        ar_utils.plot_pacf(train_set['response_body_len'])
        ar_utils.plot_acf(train_set['response_body_len'])

    t0 = times.time()
    cpu_t0 = psutil.cpu_percent(interval=0.5)
    model = ar_utils.generate_model(train_set, 0, 3, 3, 'response_body_len', algo)
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


def task5(train_t, pred_t, cpu_train, cpu_pred):
    ar_utils.plot_bar_chart(['MLE', 'PML', 'LAPLACE', 'BBVI'], train_t, 'figures/TrainingTime.jpg')
    ar_utils.plot_bar_chart(['MLE', 'PML', 'LAPLACE', 'BBVI'], pred_t, 'figures/PredictionTime.jpg')
    ar_utils.plot_bar_chart(['MLE', 'PML', 'LAPLACE', 'BBVI'], cpu_train, 'figures/CPUTraining_Perc.jpg')
    ar_utils.plot_bar_chart(['MLE', 'PML', 'LAPLACE', 'BBVI'], cpu_pred, 'figures/CPUPrediction_Perc.jpg')


def run():
    train_set, test_set = task1()
    train_mle_t, pred_mle_t, cpu_train_mle_perc, cpu_pred_mle_perc = task3(train_set, test_set, 'MLE')

    train_pml_t, pred_pml_t, cpu_train_pml_perc, cpu_pred_pml_perc = task3(train_set, test_set, 'PML')

    train_lap_t, pred_lap_t, cpu_train_lap_perc, cpu_pred_lap_perc = task3(train_set, test_set, 'Laplace')

    train_bbvi_t, pred_bbvi_t, cpu_train_bbvi_perc, cpu_pred_bbvi_perc = task3(train_set, test_set, 'BBVI')

    train_time = [train_mle_t, train_pml_t, train_lap_t, train_bbvi_t]
    pred_t = [pred_mle_t, pred_pml_t, pred_lap_t, pred_bbvi_t]

    cpu_train = [cpu_train_mle_perc, cpu_train_pml_perc, cpu_train_lap_perc, cpu_train_bbvi_perc]
    cpu_pred = [cpu_pred_mle_perc, cpu_pred_pml_perc, cpu_pred_lap_perc, cpu_pred_bbvi_perc]

    task5(train_time, pred_t, cpu_train, cpu_pred)


run()

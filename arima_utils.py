
#https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html
from statsmodels.tsa.stattools import adfuller
import pyflux as pf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np


def test_dataset(dataset):
    from pmdarima.arima.utils import ndiffs
    y = dataset.to_numpy()

    # Perform a test of stationarity for different levels of d to estimate the number of differences required to make a given time series stationary.
    ## Adf Test
    print('adf=',ndiffs(y, test='adf'))  # 0

    # KPSS test
    print('kpss=',ndiffs(y, test='kpss'))  # 1

    # PP test:
    print('pp=',ndiffs(y, test='pp'))  # 0


def adfuller_test(data):
    result=adfuller(data)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )

    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data is stationary")
    else:
        print("weak evidence against null hypothesis,indicating it is non-stationary ")

    print('\n')


def plot_acf(data):
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(data.dropna(), lags=40, ax=ax1)
    plt.title('Autocorrelation - Definition of p term - AR')
    plt.savefig('anomalous_detection_NASA/figures/acf.jpg')
    plt.clf()
    plt.cla()
    plt.close()

def plot_pacf(data):
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_pacf(data.dropna(), lags=40, ax=ax1)
    plt.title('Partial Autocorrelation - Definition of q term - MA')

    plt.savefig('anomalous_detection_NASA/figures/pacf.jpg')
    plt.clf()
    plt.cla()
    plt.close()


def plot_series (df, serie_type):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = ax1.plot(df); ax1.set_title(serie_type)
    name = 'anomalous_detection_NASA/figures/series_'+ serie_type+'.jpg'
    plt.savefig(name)
    plt.clf()
    plt.cla()
    plt.close()

'''
def print_density_residuals(model_fit):
    import pandas as pd
    model_fit.resid
    residuals = pd.DataFrame(model_fit.resid)
    fig, ax = plt.subplots(1, 2)
    residuals.plot(title="Residuals", ax=ax[0])
    residuals.plot(kind='kde', title='Density', ax=ax[1])
    plt.show()
'''

def generate_model(dataset, diff, p, q, target_field, algo):
    # Define the model
    print('generating arima model...')
    model = pf.ARIMA(data=dataset,
                     ar=p, ma=q, integ=diff, target=target_field,family=pf.Normal())

    print('training model...')
    result = model.fit(algo,nsims=50000)
    print(result.summary())
    #model.plot_fit(figsize=(15, 6))
    plt.clf()
    plt.cla()
    plt.close()

    return model

def predict (model, number_samples):
    pred_set = model.predict(h=number_samples)
    model.plot_predict(h=number_samples, figsize=(15, 5))
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()
    return pred_set


def plot_difference_model(test_set, pred_set, algo):
    # plot two lines
    print(pred_set.head(20))
    plt.plot(test_set.index, test_set.values, 'g')
    plt.plot(pred_set.index, pred_set.values, 'b')
    # set axis titles
    plt.xlabel("Timestamp")
    plt.ylabel("Response_body_len")
    # set chart title
    plt.title("Anomalous Detect Growth")
    # legend
    plt.legend(['value', 'prediction'])

    plt.savefig('anomalous_detection_NASA/figures/print_compar'+ algo+'.jpg')


def plot_bar_chart(x,y, title):
    import matplotlib.pyplot as plt
    plt.bar(x, y,color=['firebrick', 'green', 'blue', 'black'])
    plt.xlabel('Categories')
    plt.ylabel("Values")
    plt.title(title)
    plt.show()





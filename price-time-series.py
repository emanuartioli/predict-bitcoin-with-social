import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

df = pd.read_csv('C:/Users/emanu/Desktop/Thesis Project/data/price_thresholds.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# test last 3 months
train = df[df.index < '05/01/2021']
test = df[df.index >= '05/01/2021']

# price modeling
plt.plot(train['Open'])
plt.plot(test['Open'])
plt.legend(['train', 'test'])
plt.show()

# to generate the arima, we need to identify a polynomial trend
trend_coefs = np.poly1d(np.polyfit(range(len(train['Open'])), train['Open'], deg=3))
trend_line = pd.Series(trend_coefs(range(len(train['Open']))), train.index)
plt.plot(train['Open'])
plt.plot(trend_line)
plt.show()

fig, (ax1, ax2) = plt.subplots(2, 1)
plot_acf(train['Open'], ax=ax1, lags=30, zero=False, auto_ylims=True)
plot_pacf(train['Open'], ax=ax2, lags=30, zero=False, auto_ylims=True)
plt.show()

# index needs a frequency to be considered by ARIMA
df = df.asfreq('D')

# redefine train and test to include frequency
train = df[df.index < '05/01/2021']
test = df[df.index >= '05/01/2021']

ar1 = ARIMA(train['Open'], order=(1, 0, 0), trend=trend_coefs)
ar1 = ar1.fit()
print(ar1.summary())

fig, (ax1, ax2) = plt.subplots(2, 1)
plot_acf(ar1.resid, ax=ax1, lags=30, zero=False, auto_ylims=True)
plot_pacf(ar1.resid, ax=ax2, lags=30, zero=False, auto_ylims=True)
plt.show()

# after a difference there seems to be some seasonality at period 10
arima = ARIMA(train['Open'], order=(1, 0, 0), seasonal_order=(1, 0, 1, 10), trend=trend_coefs)
arima = arima.fit()
print(arima.summary())

fig, (ax1, ax2) = plt.subplots(2, 1)
plot_acf(ar1.resid, ax=ax1, lags=30, zero=False, auto_ylims=True)
plot_pacf(ar1.resid, ax=ax2, lags=30, zero=False, auto_ylims=True)
plt.show()

# the autocorrelation is also consistent with white noise


# as expected, forecast is grossly ineffective
predictions = pd.Series(ar1.forecast(len(log_test)).values, index=log_test.index)

plt.plot(np.exp(log_train))
plt.plot(np.exp(log_test))
plt.plot(np.exp(predictions))
plt.legend(['train', 'test', 'predictions'])
plt.show()
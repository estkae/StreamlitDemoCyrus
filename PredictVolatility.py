#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yfinance as yf
from arch import arch_model
from arch.__future__ import reindexing
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import date
from dateutil.utils import today
import streamlit as st
# get_ipython().run_line_magic('matplotlib', 'inline')

def tickerpv (tickertxt):
    ticker = tickertxt
    today = date.today ().strftime ("%Y-%m-%d")
    # print(today)
    # print(ticker)
    stock_data = yf.download(ticker, start="2016-01-04", end=today)

    st.table(stock_data.head())

    stock_data['Return'] = 100 * (stock_data['Close'].pct_change())

    stock_data.dropna(inplace=True)

    fig = plt.figure()
    fig.set_figwidth(12)
    plt.plot(stock_data['Return'], label = 'Daily Returns')
    plt.legend(loc='upper right')
    plt.title('Daily Returns Over Time')
    st.pyplot (plt , use_container_width=True)
    # plt.show()


    daily_volatility = stock_data['Return'].std()
    textout = ('Daily volatility: ', '{:.2f}%'.format(daily_volatility))
    st.text(textout)

    monthly_volatility = math.sqrt(21) * daily_volatility
    textout = 'Monthly volatility: ', '{:.2f}%'.format(monthly_volatility)
    st.text(textout)

    annual_volatility = math.sqrt(252) * daily_volatility
    textout = 'Annual volatility: ', '{:.2f}%'.format(annual_volatility )
    st.text (textout)


    garch_model = arch_model(stock_data['Return'], p = 1, q = 1,
                          mean = 'constant', vol = 'GARCH', dist = 'normal')

    gm_result = garch_model.fit(disp='off')
    st.text(gm_result.params)

    print('\n')

    gm_forecast = gm_result.forecast(horizon = 5)
    st.text(gm_forecast.variance[-1:])


    rolling_predictions = []
    test_size = 365

    for i in range(test_size):
        train = stock_data['Return'][:-(test_size-i)]
        model = arch_model(train, p=1, q=1)
        model_fit = model.fit(disp='off')
        pred = model_fit.forecast(horizon=1)
        rolling_predictions.append(np.sqrt(pred.variance.values[-1,:][0]))

    rolling_predictions = pd.Series(rolling_predictions, index=stock_data['Return'].index[-365:])

    plt.figure(figsize=(10,4))
    plt.plot(rolling_predictions)
    plt.title('Rolling Prediction')
    st.pyplot (plt , use_container_width=True)
    # plt.show()

    plt.figure(figsize=(12,4))
    plt.plot(stock_data['Return'][-365:])
    plt.plot(rolling_predictions)
    plt.title('Volatility Prediction - Rolling Forecast')
    plt.legend(['True Daily Returns', 'Predicted Volatility'])
    st.pyplot (plt , use_container_width=True)
    # plt.show()


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
    startdatum = date(2020,9,30)
    enddatum = date.today()
    # print (enddatum)
    totalday = enddatum-startdatum
    # print(totalday)
    # print(ticker)
    stock_data = yf.download(ticker, start=startdatum, end=enddatum)
    print("Info: ",len(stock_data.index))

    st.table(stock_data.tail())
    st.table(stock_data.head())
    # st.table(stock_data.count())
    if totalday.days > 365.0:
        totalday = 365
    else:
        # totalday = stock_data.count().iloc[1:1]
        totalday = len(stock_data.index)-5

    #print ("TotalTage: ",stock_data.iloc[2:1])
    stock_data['Return'] = 100 * (stock_data['Close'].pct_change())

    # st.table (stock_data.tail ())

    stock_data.dropna(inplace=True)

    # st.table (stock_data.tail ())


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

    gm_forecast = gm_result.forecast(horizon = 5)
    st.text(gm_forecast.variance[-1:])


    rolling_predictions = []

    test_size = totalday
    #print ("Rolling",test_size.days)

    for i in range(test_size):
        # print(stock_data.count())
        train = stock_data['Return'][:-(test_size-i)]
        # print("Index",i,"TS", test_size, train.head())
        model = arch_model(train, p=1, q=1)
        model_fit = model.fit(first_obs=startdatum, last_obs=enddatum, disp='off')
        pred = model_fit.forecast(horizon=1)
        rolling_predictions.append(np.sqrt(pred.variance.values[-1,:][0]))
    # print (pred.variance.tail ())
    rolling_predictions = pd.Series(rolling_predictions, index=stock_data['Return'].index[-totalday:])

    plt.figure(figsize=(10,4))
    plt.plot(rolling_predictions)
    plt.title('Rolling Prediction')
    st.pyplot (plt , use_container_width=True)
    # plt.show()

    plt.figure(figsize=(12,4))
    plt.plot(stock_data['Return'][-totalday:])
    plt.plot(rolling_predictions)
    plt.title('Volatility Prediction - Rolling Forecast')
    plt.legend(['True Daily Returns', 'Predicted Volatility'])
    st.pyplot (plt , use_container_width=True)
    # plt.show()


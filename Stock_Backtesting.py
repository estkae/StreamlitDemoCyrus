#!/usr/bin/env python
# coding: utf-8

# In[8]:


import talib
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import date
import pandas as pd
import bt
import os
import streamlit as st



def ticker (tickertxt):
    ticker = tickertxt
    today = date.today ().strftime ("%Y-%m-%d")
    stock_data = yf.download (ticker , start="2020-1-1" , end=today)
    st.table(stock_data.tail())

    EMA_short = talib.EMA (stock_data['Close'] , timeperiod=12).to_frame ()
    EMA_short = EMA_short.rename (columns={0: 'Close'})
    EMA_long = talib.EMA (stock_data['Close'] , timeperiod=50).to_frame ()
    EMA_long = EMA_long.rename (columns={0: 'Close'})

    signal = EMA_long.copy ()
    signal[EMA_long.isnull ()] = 0
    signal[EMA_short > EMA_long] = 1
    signal[EMA_short < EMA_long] = -1

    print(type(signal))

    st.markdown(signal[50:250])

    transition = signal[signal['Close'].diff() != 0]
    buy_signal = transition[transition['Close'] == 1]
    sell_signal = transition[transition['Close'] == -1]

    buy_index = buy_signal.index
    buy_position = stock_data[stock_data.index.isin(buy_index)]
    sell_index = sell_signal.index
    sell_position = stock_data[stock_data.index.isin(sell_index)]

    fig = go.Figure()
    fig.add_trace(
            go.Candlestick(x=stock_data.index,
                            open=stock_data['Open'],
                            high=stock_data['High'],
                            low=stock_data['Low'],
                            close=stock_data['Close'],
                            name="Stock Prices"
                          )
    )

    fig.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=EMA_long['Close'],
                name="EMA 50"
            )
    )

    fig.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=EMA_short['Close'],
                name = "EMA 12"
            )
    )

    fig.add_trace(
            go.Scatter(
                x=buy_position.index,
                y=buy_position['Close'],
                name="Buy Signal",
                marker=dict(color="#511CFB", size=15),
                mode="markers",
                marker_symbol="triangle-up"
            )
    )

    fig.add_trace(
            go.Scatter(
                x=sell_position.index,
                y=sell_position['Close'],
                name="Sell Signal",
                marker=dict(color="#750086", size=15),
                mode="markers",
                marker_symbol="triangle-down"
            )
    )

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        title="Daily Close (" + ticker + ") Prices",
        xaxis_title="Date",
        yaxis_title="Price (USD)"
    )
    st.plotly_chart (fig , use_container_width=True)
    #
    #
    # #get_ipython().run_line_magic('matplotlib', 'inline')
    #
    bt_strategy = bt.Strategy('EMA_crossover',
                                [   bt.algos.RunWeekly(),
                                    bt.algos.WeighTarget(signal),
                                    bt.algos.Rebalance()
                                ]
                             )

    bt_backtest = bt.Backtest (bt_strategy , stock_data['Close'].to_frame ())
    bt_result = bt.run (bt_backtest)
    bt_result.plot (title='Backtest result (Equity Progression)')
    st.pyplot (plt , use_container_width=True)


    bt_result.plot_histograms(bins=50, freq = 'w')
    st.pyplot (plt , use_container_width=True)

    #
    # print("Type",type(bt_result.display()))
    # print(bt_result.to_csv().replace(",,,,,,,,,,,,,,,,,",","))
    cwd =os.getcwd()
    data = bt_result.to_csv(cwd+"/blabla.csv").replace(",,,,,,,,,,,,,,,,,",",")
    # print(data)
    # print(pd.read_csv(data.    split(sep="\n")))
    # bt_result.to_csv(sep=";",path="/Users/karlestermann/PycharmProjects/StreamlitDemoCyrus/blabla.csv").replace(",,,,,,,,,,,,,,,,,",",")
    data1 = pd.read_csv(cwd+"/blabla.csv",sep=",",skip_blank_lines=True,na_values=None)
    # print(data1)
    st.table(data1)


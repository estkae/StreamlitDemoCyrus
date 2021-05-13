#!/usr/bin/env python
# coding: utf-8

#Description: This program uses the Bollinger Band strategy to determine when to buy and sell stock


#Import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
plt.style.use('fivethirtyeight')
import yfinance as yf
import streamlit as st

#Store the data
def tickerbol(tickertxt):
    ticker = tickertxt
    today = date.today ().strftime ("%Y-%m-%d")
    df = yf.download(ticker, start="2016-01-04", end=today)
    st.table(df.tail())
    #print(df['Date'])
    # df = df.set_index(pd.DatetimeIndex(df['Date'].values))
    # #Show the data

    #Graph the data
    #Get the time period (20 days)
    period = 20
    # Calculate the 20 Day Simple Moving Average, Std Deviation, Upper Band and Lower Band
    #Calculating the Simple Moving Average
    df['SMA'] = df['Close'].rolling(window=period).mean()
    # Get the standard deviation
    df['STD'] = df['Close'].rolling(window=period).std()
    #Calculate the Upper Bollinger Band
    df['Upper'] = df['SMA'] + (df['STD'] * 2)
    #Calculate the Lower Bollinger Band
    df['Lower'] = df['SMA'] - (df['STD'] * 2)
    #Create a list of columns to keep
    column_list = ['Close', 'SMA', 'Upper', 'Lower']
    df[column_list].plot(figsize=(12.2,6.4)) #Plot the data
    print(df)
    plt.title('Bollinger Band for '+tickertxt)
    plt.ylabel('USD Price ($)')
    st.pyplot (plt , use_container_width=True)
    # plt.show();


    #Plot and shade the area between the two Bollinger bands
    #Get the figure and the figure size
    fig = plt.figure(figsize=(12.2,6.4)) #width = 12.2 inches and height = 6.4 inches
    #Add the subplot
    ax = fig.add_subplot(1,1,1) #Number of rows, cols, & index
    # Get the index values of the DataFrame
    x_axis = df.index
    # Plot and shade the area between the upper band and the lower band Grey
    ax.fill_between(x_axis, df['Upper'], df['Lower'], color='grey')
    # Plot the Closing Price and Moving Average
    ax.plot(x_axis, df['Close'], color='gold', lw=3, label = 'Close Price') #lw = line width
    ax.plot(x_axis, df['SMA'], color='blue', lw=3, label = 'Simple Moving Average')
    # Set the Title & Show the Image
    ax.set_title('Bollinger Band For '+tickertxt)
    ax.set_xlabel('Date')
    ax.set_ylabel('USD Price ($)')
    plt.xticks(rotation = 45)
    ax.legend()
    st.pyplot (plt , use_container_width=True)
    # plt.show();

    #Create a new data frame
    new_df = df[period-1:]
    print (new_df)

    # Create a function to get the buy and sell signals
    def get_signal(data):
      buy_signal = [] #buy list
      sell_signal = [] #sell list
      for i in range(len(data['Close'])):
        if data['Close'][i] > data['Upper'][i]: #Then you should sell
          #print('SELL')
          buy_signal.append(np.nan)
          sell_signal.append(data['Close'][i])
        elif data['Close'][i] < data['Lower'][i]: #Then you should buy
          #print('BUY')
          sell_signal.append(np.nan)
          buy_signal.append(data['Close'][i])
        else:
          buy_signal.append(np.nan)
          sell_signal.append(np.nan)
      return (buy_signal, sell_signal)

    new_df['Buy'] =  get_signal(new_df)[0]
    new_df['Sell'] =  get_signal(new_df)[1]

    #Plot all of the data
    #Get the figure and the figure size
    fig = plt.figure(figsize=(12.2,6.4)) #width = 12.2 inches and height = 6.4 inches
    #Add the subplot
    ax = fig.add_subplot(1,1,1) #Number of rows, cols, & index
    # Get the index values of the DataFrame
    x_axis = new_df.index
    # Plot and shade the area between the upper band and the lower band Grey
    ax.fill_between(x_axis, new_df['Upper'], new_df['Lower'], color='grey')
    # Plot the Closing Price and Moving Average
    ax.plot(x_axis, new_df['Close'], color='gold', lw=3, label = 'Close Price',alpha = 0.5)
    ax.plot(x_axis, new_df['SMA'], color='blue', lw=3, label = 'Moving Average',alpha = 0.5)
    ax.scatter(x_axis, new_df['Buy'] , color='green', lw=3, label = 'Buy',marker = '^', alpha = 1)
    ax.scatter(x_axis, new_df['Sell'] , color='red', lw=3, label = 'Sell',marker = 'v', alpha = 1)
    # Set the Title & Show the Image
    ax.set_title('Bollinger Band For '+tickertxt)
    ax.set_xlabel('Date')
    ax.set_ylabel('USD Price ($)')
    plt.xticks(rotation = 45)
    ax.legend()
    st.pyplot (plt , use_container_width=True)
    # plt.show();

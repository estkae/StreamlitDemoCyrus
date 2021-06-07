#!/usr/bin/env python
# coding: utf-8

# <h1 align="center">Stock market screening and analysis</h1>

# <font size="3">Written by Ibinabo Bestmann</font>

# Stock markets tend to react very quickly to a variety of factors such as news, earnings reports, etc. While it may be prudent to develop trading strategies based on fundamental data, the rapid changes in the stock market are incredibly hard to predict and may not conform to the goals of more short term traders. This study aims to use data science as a means to both identify high potential stocks, as well as attempt to forecast future prices/price movement in an attempt to maximize an investor’s chances of success.
#
# In the first half of this analysis, I will introduce a strategy to search for stocks that involves identifying the highest-ranked stocks based on trading volume during the trading day. I will also include information based on twitter and sentiment analysis in order to provide an idea of which stocks have the maximum probability of going up in the near future. The next half of the project will attempt to apply forecasting techniques to our chosen stock(s). I will apply deep learning via a **Long short term memory (LSTM) neural network**, which is a form of a recurrent neural network (RNN) to predict close prices. Finally, I will also demonstrate how **simple linear regression** could aid in forecasting.

# ## Part 1: Stock screening
#
# Let's begin by importing relevant packages

# In[56]:


import yahoo_fin.stock_info as ya
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.sectorperformance import SectorPerformances
import pandas as pd
import pandas_datareader as web
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import requests
import time
import numpy as np
import seaborn as sns
import mplfinance as mpf
from tensorflow import keras
import streamlit as st
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import median_absolute_error, mean_squared_error

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)


def plot_learningCurve(history,epoch_num):

    epoch_range = range(1,epoch_num+1)

    plt.plot(epoch_range,history.history['loss'])
    plt.plot(epoch_range,history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train','val'], loc='upper left')
    st.pyplot (plt , use_container_width=True)
    #plt.show()


def predict_range(X,y,model,conf=2.58):

    from numpy import sum as arraysum

    # Obtain predictions
    yhat = model.predict(X)

    # Compute standard deviation
    sum_errs = arraysum((y - yhat)**2)
    stdev = np.sqrt(1/(len(y)-2) * sum_errs)

    interval = conf * stdev

    lower = []
    upper = []

    for i in yhat:
        lower.append(i-interval)
        upper.append(i+interval)


    return lower, upper, interval



# We begin by scrape data on the most active stocks in a given time period, in this case one day. Higher trading volume is more likely to result in bigger price volatility which could potentially result in larger gains.

# In[57]:

def stockmarket(tickertxt):

    movers = ya.get_day_most_active()
    movers.head()


# Right away we notice that stocks with negative price changes are also included in our results. A filter to get only stocks with a positive % change is applied to get our desired stocks

# In[58]:


    movers = movers[movers['% Change'] >= 0]
    movers.head()


# Excellent! We have successfully scraped the data using the yahoo_fin python module. it is often a good idea to see if those stocks are also generating attention, and what kind of attention it is to avoid getting into false rallies. We will scrap some sentiment data courtesty of [sentdex](http://www.sentdex.com/financial-analysis/). Sometimes sentiments may lag due to source e.g Newsarticle published an hour after event, so we will also utilize [tradefollowers](https://www.tradefollowers.com/strength/twitter_strongest.jsp?tf=1d) for their twitter sentiment data. We will process both lists independently and combine them. For both the sentdex and tradefollowers data we use a 30 day time period. Using a single day might be great for day trading but increases probability of jumping on false rallies.
#
# NOTE: Sentdex only has stocks which belong to the S&P 500

# In[59]:


    res = requests.get('http://www.sentdex.com/financial-analysis/?tf=30d')
    soup = BeautifulSoup(res.text)
    table = soup.find_all('tr')


# In[60]:


    stock = []
    sentiment = []
    mentions = []
    sentiment_trend = []

    for ticker in table:
        ticker_info = ticker.find_all('td')

        try:
            stock.append(ticker_info[0].get_text())
        except:
            stock.append(None)
        try:
            sentiment.append(ticker_info[3].get_text())
        except:
            sentiment.append(None)
        try:
            mentions.append(ticker_info[2].get_text())
        except:
            mentions.append(None)
        try:
            if (ticker_info[4].find('span',{"class":"glyphicon glyphicon-chevron-up"})):
                sentiment_trend.append('up')
            else:
                sentiment_trend.append('down')
        except:
            sentiment_trend.append(None)


    company_info = pd.DataFrame(data={'Symbol': stock, 'Sentiment': sentiment, 'direction': sentiment_trend, 'Mentions':mentions})

    company_info


# We then combine these results with our results from the biggest movers on a given day. This done using a left join of this data frame with the original movers data frame

# In[61]:


    top_stocks = movers.merge(company_info, on='Symbol', how='left')
    top_stocks.drop(['Market Cap','PE Ratio (TTM)'], axis=1, inplace=True)
    top_stocks


# A couple of stocks pop up with both very good sentiments and an upwards trend in favourability. ZNGA, TWTR and AES for instance stood out as potentially good picks. Note, the mentions here refer to the number of times the stock was referenced according to the internal metrics used by [sentdex](sentdex.com). Let's attempt supplimenting this information with some data based on twitter. We get stocks that showed the strongest twitter sentiments with a time period of 1 month

# In[62]:


    res = requests.get("https://www.tradefollowers.com/strength/twitter_strongest.jsp?tf=1m")
    soup = BeautifulSoup(res.text)

    stock_twitter = soup.find_all('tr')


# In[63]:


    twit_stock = []
    sector = []
    twit_score = []

    for stock in stock_twitter:
        try:
            score = stock.find_all("td",{"class": "datalistcolumn"})
            twit_stock.append(score[0].get_text().replace('$','').strip())
            sector.append(score[2].get_text().replace('\n','').strip())
            twit_score.append(score[4].get_text().replace('\n','').strip())
        except:
            twit_stock.append(np.nan)
            sector.append(np.nan)
            twit_score.append(np.nan)

    twitter_df = pd.DataFrame({'Symbol': twit_stock, 'Sector': sector, 'Twit_Bull_score': twit_score})

# Remove NA values
    twitter_df.dropna(inplace=True)
    twitter_df.drop_duplicates(subset ="Symbol",
                         keep = 'first', inplace = True)
    twitter_df.reset_index(drop=True,inplace=True)
    twitter_df


# Twit_Bull_score refers to the internally scoring used at [tradefollowers](tradefollowers.com) to rank stocks based on twitter sentiments, and can range from 1 to as high as 10,000 or greater. With the twitter sentiments obtains, we combine it with our sentiment data to get an overall idea of the data.

# In[64]:


    Final_list =  top_stocks.merge(twitter_df, on='Symbol', how='left')
    Final_list


# Finally, we include a twitter momentum score.

# In[65]:


    res2 = requests.get("https://www.tradefollowers.com/active/twitter_active.jsp?tf=1m")
    soup2 = BeautifulSoup(res2.text)

    stock_twitter2 = soup2.find_all('tr')


# In[66]:


    twit_stock2 = []
    sector2 = []
    twit_score2 = []

    for stock in stock_twitter2:
        try:
            score2 = stock.find_all("td",{"class": "datalistcolumn"})



            twit_stock2.append(score2[0].get_text().replace('$','').strip())
            sector2.append(score2[2].get_text().replace('\n','').strip())
            twit_score2.append(score2[4].get_text().replace('\n','').strip())
        except:
            twit_stock2.append(np.nan)
            sector2.append(np.nan)
            twit_score2.append(np.nan)

    twitter_df2 = pd.DataFrame({'Symbol': twit_stock2, 'Sector': sector2, 'Twit_mom': twit_score2})

    # Remove NA values
    twitter_df2.dropna(inplace=True)
    twitter_df2.drop_duplicates(subset ="Symbol",
                         keep = 'first', inplace = True)
    twitter_df2.reset_index(drop=True,inplace=True)
    twitter_df2


# We again combine the dataframes to earlier concatanated dataframes. This will form our recommender list

# In[67]:


    Recommender_list = Final_list.merge(twitter_df2, on='Symbol', how='left')
    Recommender_list.drop(['Volume','Avg Vol (3 month)'],axis=1, inplace=True)
    Recommender_list


# Our list now contains even more informationt to help us with our trades. Stocks which it suggests might generate positive returns include TSLA, ZNGA and TWTR. There is also the posibility that we do not get a stock that falls in all our generated lists, so usage of, for instance, the price information and the twitter data could still give us a good idea of what to expect in terms of performance. As an added measure, we can also obtain information on the sectors to see how they've performed. Again, we will use a one month time period for comparison. The aforementioned stocks belong to the Technology and consumer staples sectors.

# In[68]:


    sp = SectorPerformances(key='0E66O7ZP6W7A1LC9', output_format='pandas')
    plt.figure(figsize=(8,8))
    data, meta_data = sp.get_sector()
    st.text(meta_data)
    data['Rank D: Month Performance'].plot(kind='bar')
    plt.title('One Month Performance (%) per Sector')
    plt.tight_layout()
    plt.grid()
    st.pyplot (plt , use_container_width=True)
    #plt.show()


# The industrials sector appears to be the best performing in this time period. Consumer staples appears to be doing better than IT, but overall they are up which bodes well for potential investors. Please note that this analysis is only a guide to find potentially positive return generating stocks. It is still up to the investor to do the research.

# ## Part 2: Forecasting using an LSTM
#
# In this section, we will atetmpt to apply deep learning to a stock of our chosing to predict future prices. At the time this project was conceived, the stock AMD was selected as it experienced really high gains at the time.

# First we obtain stock data for our chosen stock. Data from 2014 data up till August of 2020 was obtained for our analysis. Our data will be obtained from yahoo

# In[69]:


    from datetime import datetime
    from datetime import date

    today = date.today()
    #today.replace("-",",")
    #print(today)


# In[70]:


    start = datetime(2014,12,31)
    end = datetime(2021,6,3)
    #print(end)


# In[71]:


    stock_dt = web.DataReader('AMD','yahoo',start,end)
    stock_dt.reset_index(inplace=True)
    stock_dt.head()


# In[72]:


    stock_dt.tail()


# ### Feature selection/engineering
#
# We add additional data that might potentially increase prediction accuracy. Here we use technical indicators.

# In[73]:


# Technical Indicators

    # RSI
    t_rsi = TechIndicators(key='0E66O7ZP6W7A1LC9',output_format='pandas')
    data_rsi, meta_data_rsi = t_rsi.get_rsi(symbol='AMD', interval='daily',time_period = 9,
                              series_type='open')
    timeout = 15

    # SMA
    t_sma = TechIndicators(key='0E66O7ZP6W7A1LC9',output_format='pandas')
    data_sma, meta_data_sma = t_sma.get_sma(symbol='AMD', interval='daily',time_period = 9,
                              series_type='open')
    timeout = 15

    #EMA
    t_ema = TechIndicators(key='0E66O7ZP6W7A1LC9',output_format='pandas')
    data_ema, meta_data_ema = t_ema.get_ema(symbol='AMD', interval='daily',time_period = 9,
                              series_type='open')
    timeout = 15


# In[74]:


    #On Balance volume
    t_obv = TechIndicators(key='0E66O7ZP6W7A1LC9',output_format='pandas')
    data_obv, meta_data_obv = t_obv.get_obv(symbol='AMD', interval='daily')
    timeout = 15

    # Bollinger bands
    t_bbands = TechIndicators(key='0E66O7ZP6W7A1LC9',output_format='pandas')
    data_bbands, meta_data_bb = t_bbands.get_bbands(symbol='AMD', interval='daily', series_type='open', time_period=9)


# To learn more about technical indicators and how they are useful in stock analysis, I welcome you to explore [investopedia](https://www.investopedia.com/). Let's combine these indicators into a dataframe

# In[75]:


    t_ind = pd.concat([data_ema, data_sma, data_rsi, data_obv, data_bbands],axis=1)
    t_ind


# We then extract the values for the time interval of choice

# In[76]:


    t_ind = t_ind.loc[start:end].reset_index()


# Now we combine them with our original dataframe containing price and volume information

# In[77]:


    df_updated = pd.concat([stock_dt,t_ind],axis=1)
    df_updated.set_index('Date',drop=True,inplace=True)
    df_updated


# Before we begin, it is often a good idea to visually inspect the stock data to have an idea of the price trend and volume information

# In[78]:




# In[79]:


    mpf.plot(df_updated.loc[datetime(2020,5,1):datetime(2021,6,3)],type='candle',style='yahoo',figsize=(8,6),volume=True)


# in the month of July, AMD experienced a massive price surge. Let's have a look at the data with the indicators included

# In[80]:


    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(12,12))

    ax[0].plot(df_updated['Open'].loc[datetime(2020,5,1):datetime(2020,8,11)],'k',lw=2,label='Close')
    ax[0].plot(df_updated['EMA'].loc[datetime(2020,5,1):datetime(2020,8,11)],'r',lw=1.5,label='EMA')
    ax[0].plot(df_updated['SMA'].loc[datetime(2020,5,1):datetime(2020,8,11)],'b',lw=1.5,label='SMA')
    ax[0].plot(df_updated['Real Upper Band'].loc[datetime(2020,5,1):datetime(2020,8,11)],'g',lw=1.5,label='Boolinger band (upper)')
    ax[0].plot(df_updated['Real Lower Band'].loc[datetime(2020,5,1):datetime(2020,8,11)],'y',lw=1.5,label='Boolinger band (lower)')
    ax[0].set_ylabel('Closing price')


    ax[0].legend()


    temp = len(df_updated['RSI'].loc[datetime(2020,5,1):datetime(2020,8,11)])

    ax[1].plot(df_updated['RSI'].loc[datetime(2020,5,1):datetime(2020,8,11)],'g',lw=2,label='RSI')
    ax[1].plot(df_updated['RSI'].loc[datetime(2020,5,1):datetime(2020,8,11)].index,70*np.ones((temp,1)).flatten(),'k')
    ax[1].plot(df_updated['RSI'].loc[datetime(2020,5,1):datetime(2020,8,11)].index,30*np.ones((temp,1)).flatten(),'k')
    ax[1].set_ylabel('RSI')
    #ax[1].legend()

    ax[2].plot(df_updated['OBV'].loc[datetime(2020,5,1):datetime(2020,8,11)],'y',lw=2,label='OBV')
    ax[2].set_ylabel('On balance Volume')
    #ax[2].legend()
    ax[2].set_xlabel('Date');
    st.pyplot (fig)



# Indicators give us an idea of the direction of future prices. For instance, the Exponential moving average (EMA) crossing the Simple moving average (SMA) might indicate a positive uptrend in price. RSI gives us an idea of how much the stock is being bought or sold. An RSI of 70 for instance might indicate an overbought stock, and tells us the price is very likely to go down in the future, while an RSI of 30 indicates an oversold stock and could potentially be a good buy point for a stock. On balance volume gives us the relative changes in volume, and can potentially identify true rallies or breakouts. Bollinger bands provide an idea of the volatility of the stock.
#
# We also want to take into account relative changes between trading days as they tend to be less volatile, and therefore a bit more stationary. We will take the difference between two consecutive days in this case.

# In[81]:


    df_updated['Diff_Open'] = df_updated['Open'] - df_updated['Open'].shift(1)
    df_updated['Diff_Close'] = df_updated['Close'] - df_updated['Close'].shift(1)
    df_updated['Diff-Volume']  = df_updated['Volume'] - df_updated['Volume'].shift(1)
    df_updated['Diff-High']  = df_updated['High'] - df_updated['High'].shift(1)
    df_updated['Diff-Low']  = df_updated['Low'] - df_updated['Low'].shift(1)
    df_updated['Diff-Close (forward)'] = np.where(df_updated['Close'].shift(-1) > df_updated['Close'],1,-1)


    df_updated['High-Low'] = df_updated['High'] - df_updated['Low'].shift(1)
    df_updated['Open-Close'] = df_updated['Open'] - df_updated['Close'].shift(1)

    df_updated['Returns'] = df_updated['Open'].pct_change(1)


# In[82]:


    df_updated.head()


# The next step is to visualize how the features relate to each other. We employ a correlation matrix for this purpose

# In[83]:


    df_updated.drop(['date','Real Middle Band','Adj Close'],axis=1,inplace=True)


# In[84]:



    plt.figure(figsize=(12,8))
    sns.heatmap(df_updated.corr());


# The closing price has very strong correlations with some of the other price informations such as opening price, highs and lows.
# On the other hands, the differential prices arn't as correlated. We want to limit the amount of colinearity in our system before running any machine learning routine. So feature selection is a must.

# ### Feature Selection
#
# We utilize two means of feature selection in this section. Random forests and mutual information gain. Random forests are
# very popular due to their relatively good accuracy, robustness as well as simplicity in terms of utilization. They can directly measure the impact of each feature on accuracy of the model and in essence give them a rank. Information gain on the other hand, calculates the reduction in entropy from transforming a dataset in some way. Mutual information gain essentially evaluates the gain of each variable in the context of the target variable.

# In[85]:



# ### Random forest regressor

# In[88]:


# Seperate the target variable from the features
    y = df_updated['Close'].iloc[1:].dropna()
    X = df_updated.drop(['Close'],axis=1).iloc[1:].dropna()
    #print("y-Band: ",y.count)
    #print("x-band: ",X.count)


# In[89]:


    X_train, X_test,y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)


# In[90]:


    X_train.shape, y_train.shape


# In[92]:


    feat = SelectFromModel(RandomForestRegressor(n_estimators=100,random_state=0,n_jobs=-1))
    feat.fit(X_train,y_train)
    feat.get_support()


# In[93]:


    X_train.columns[feat.get_support()]


# The regressor essentially selected the features that displayed good correlation with the Close price. However, although it selected the most important we would like information on the information gain from each variable. An issue with using random forests is it tends to diminsh the importance of other correlated variables and may lead to incorrect interpretation. However, it does help reduce overfitting

# ### Mutual information gain

# In[94]:




# In[96]:


    mi = mutual_info_regression(X_train,y_train)
    mi = pd.Series(mi)
    mi.index = X_train.columns
    mi.sort_values(ascending=False,inplace=True)


# In[97]:


    mi


# The results validate the results using the random forest regressor, but it appears some of the other variables also contribute
# a decent amount of information. We will select values greater than 2 for our analysis.

# In[98]:


    sel = SelectKBest(mutual_info_regression, k = 8).fit(X_train, y_train) #
    Features = X_train.columns[sel.get_support()]
    Features.values


# ### Preprocessing
#
# In order to construct a Long short term memory neural network (LSTM), we need to understand its structure. Below is the design of a typical LSTM unit.  Data source: [Researchgate](https://www.researchgate.net/publication/334268507_Application_of_Long_Short-Term_Memory_LSTM_Neural_Network_for_Flood_Forecasting)

# ![LSTM_structure.jpg](LSTM_structure.jpg)

# As mentioned earlier, LSTM's are a special type of Recurrent neural networks (RNN). Recurrent neural networks (RNN) are a special type of neural network in which the output of a layer is fed back to the input layer multiple times in order to learn from the past data. Basically, the neural network is trying to learn data that follows a sequence. However, since the RNNs utilize past data, they can become computationally expensive due to storing large amouts of data in memory. The LSTM mitigates this issue, using gates. It has a cell state, and 3 gates; forget, imput and output gates.
#
# The cell state is essentially the memory of the network. It carries information throughtout the data sequence processing. Information is added or removed from this cell state using gates. Information from the previous hidden state and current input are combined and passed through a sigmoid function at the forget gate. The sigmoid function determines which data to keep or forget. The transformed values are then multipled by the current cell state.
#
# Next, the information from the previous hidden state combined with the input is passed through a sigmoid function to again determine important information, and also a tanh function to transform data between -1 and 1. This transformation helps with the stability of the network and helps deal with the vanishing/exploding gradient problem. These 2 outputs are multiplied together, and the output is added to the current cell state with the sigmoid function applied to it to give us our new cell state for the next time step.
#
# Finally, the information from the hidden state combined with the current input are combined and a sigmoid function applied to it. The new cell state is passed through a tanh function to transform the values and both outputs are multiplied to determine the new hidden state for the next time step.
#
# Now we have an idea of how the LSTM works, let's construct one. First we split our data into training and test set

# In[99]:


    df_updated.reset_index(drop=True,inplace=True)

    train_size = int(len(df_updated) * 0.8)
    test_size =  len(df_updated) - train_size

# Make sure to omit the first row, contains NAN's
    train = df_updated.iloc[1:train_size]
    test = df_updated.iloc[train_size:]


# In[100]:


    train.shape, test.shape


# In[102]:


# Extract the features
    total_features = list(Features.values)

    total_features.append('Close')
    total_features


    train = train[total_features]
    test = test[total_features]

    train.shape,test.shape


# Before we proceed, it is important to scale the data. Scaling is done to ensure one set of features don't have more importance relative to the others. In addition, having values between 0 and 1 will help the neural network converge faster if at all it does. We apply different scalings to the test and training data to avoid leakage into our model.

# In[103]:


# Scale both features and target variables

    f_transformer = MinMaxScaler() # Feature scaler
    targ_transformer = MinMaxScaler() # Target scaler


    f_transformer = f_transformer.fit(train[Features].to_numpy())
    targ_transformer = targ_transformer.fit(train[['Close']])

    train.loc[:,Features] = f_transformer.transform(train[Features].to_numpy())
    train['Close'] = targ_transformer.transform(train[['Close']].to_numpy())

    test.loc[:,Features] = f_transformer.transform(test[Features].to_numpy())
    test['Close'] = targ_transformer.transform(test[['Close']].to_numpy())


# In[104]:


    train.shape, test.shape


# The figure below shows how the sequential data for an LSTM is constructed to be fed into the network. Data source: [Althelaya et al, 2018](https://ieeexplore.ieee.org/document/8355458)

# ![LSTM_data_arrangement.PNG](attachment:LSTM_data_arrangement.PNG)

# Bassically for data at time t, with a window size of N, the target feature will be the data point at time t, and the feature will be the data points [t-1, t-N]. We then sequentially move forward in time using this approach. We therefore need to format our data that way.

# In[105]:



# In[106]:


    time_steps = 10

    X_train_lstm, y_train_lstm = create_dataset(train.drop(['Close'],axis=1), train['Close'], time_steps)
    X_test_lstm, y_test_lstm = create_dataset(test.drop(['Close'],axis=1), test['Close'], time_steps)


# In[108]:


    X_train_lstm.shape, y_train_lstm.shape


# In[109]:


    X_test_lstm.shape, y_test_lstm.shape


# ### Building LSTM model
#
# The new installment of tensorflow (Tensorflow 2.0) via keras has made implmentation of deep learning models much easier than in previous installments. We will apply a bidrectional LSTM as they have been shown to more effective in certain applications (see [Althelaya et al, 2018](https://ieeexplore.ieee.org/document/8355458)). This due to the fact that the network learns using both past and future data in 2 layers. Each layer performs the operations using reversed time steps to each other. The loss function in this case will be the mean squared error, and the adam optimizer with the default learning rate is applied.

# In[110]:




# In[111]:


    model = keras.Sequential()
    model.add(keras.layers.Bidirectional(
            keras.layers.LSTM(
              units=32,
              input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]))
      )
    )

    model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.Dense(units=1))


# In[112]:


    model.compile(optimizer='adam',loss='mean_squared_error')


# In[114]:


    history = model.fit(
        X_train_lstm, y_train_lstm,
        epochs=90,
        batch_size=40,
        validation_split=0.2,
        shuffle=False,
        verbose=1
    )


# In[115]:


    test_loss = model.evaluate(X_test_lstm,y_test_lstm)


# In[116]:


# In[117]:


    plot_learningCurve(history,90)


# With each epoch, the validation loss is decreasing but in a bit of a stochastic manner. The training loss is fairly consisten throughout. There maybe some overfitting in there but you can always tune model parameters and explore data more. Let's make some predictions on the test data just to see what's happening

# In[118]:


    y_pred = model.predict(X_test_lstm)


# We need to apply some inverse scaling to get back our original results.

# In[119]:


    y_train_inv = targ_transformer.inverse_transform(y_train_lstm.reshape(1, -1))
    y_test_inv = targ_transformer.inverse_transform(y_test_lstm.reshape(1, -1))
    y_pred_inv = targ_transformer.inverse_transform(y_pred)


# In[120]:


    plt.figure(figsize=(10,10))
    plt.plot(np.arange(0, len(y_train_lstm)), y_train_inv.flatten(), 'g', label="history")
    plt.plot(np.arange(len(y_train_lstm,), len(y_train_lstm) + len(y_test_lstm)), y_test_inv.flatten(), marker='.', label="true")
    plt.plot(np.arange(len(y_train_lstm), len(y_train_lstm) + len(y_test_lstm)), y_pred_inv.flatten(), 'r', label="prediction")
    plt.ylabel('Close Price')
    plt.xlabel('Time step')
    plt.legend()
    st.pyplot (plt , use_container_width=True)
    #plt.show();


# At first glance we can see that the our predictions are not very great, we could define adjust our model parameters some more. However, they appear to be following the trends pretty well. Let's take a closer look

# In[121]:


    plt.figure(figsize=(10,10))
    plt.plot(np.arange(len(y_train_lstm[0:500],), len(y_train_lstm[0:500]) + len(y_test_lstm[0:500])), y_test_inv.flatten()[0:500],  label="true")
    plt.plot(np.arange(len(y_train_lstm[0:500]), len(y_train_lstm[0:500]) + len(y_test_lstm[0:500])), y_pred_inv.flatten()[0:500], 'r', label="prediction")
    plt.ylabel('Close Price')
    plt.xlabel('Time Step')
    plt.legend()
    st.pyplot (plt , use_container_width=True)
    #plt.show();


# Now it will become apparent why I did not use a large amount of epochs to train my model. At first glance, we notice the LSTM has some implicit autocorrelation in its results since its predictions for a given day are very similar to those of the previous day. It essentially lags. Its basically showing that the best guess of the model is very similar to previous results. This should not be a surprising result; The stock market is influenced by a number of factors such as news, earnings reports, meargers etc. Therefore, it is a bit too choatic and stoachastic to be acurately modelled because it depends on so many factors, some of which can be sporadic i.e positive or negative news. Therefore in my opinion, this may not be the best way to predict stock prices. Of course with major advances in AI there might actually be a way, but I don't think the hedge funds will be sharing their methods anytime soon.

# ## Part 3: Regression analysis

# Of course we could still make an attempt to have an idea of what the possible price movements might be. In this case I will utilize the differential prices as there's less volatility compared to using absolute prices. Let's explore these relationships

# In[122]:


    fig, ax = plt.subplots(nrows=3,ncols=2,figsize=(10,10))

    ax[0,0].scatter(df_updated['Open-Close'],df_updated['Diff_Close'],c='k')
    ax[0,0].legend(['Open-Close'])
    ax[0,0].set_ylabel('Diff-Close')

    ax[0,1].scatter(df_updated['High-Low'],df_updated['Diff_Close'],c='k')
    ax[0,1].legend(['High-Low'])
    ax[0,1].set_ylabel('Diff-Close')

    ax[1,0].scatter(df_updated['Diff_Open'],df_updated['Diff_Close'],c='k')
    ax[1,0].legend(['Diff-Open'])
    ax[1,0].set_ylabel('Diff-Close')

    ax[1,1].scatter(df_updated['Diff-Low'],df_updated['Diff_Close'],c='k')
    ax[1,1].legend(['Diff-Low'])
    ax[1,1].set_ylabel('Diff-Close')

    ax[2,0].scatter(df_updated['Diff-High'],df_updated['Diff_Close'],c='k')
    ax[2,0].legend(['Diff-High'])
    ax[2,0].set_ylabel('Diff-Close')

    ax[2,1].scatter(df_updated['Open'],df_updated['Diff_Close'],c='k')
    ax[2,1].legend(['Open'])
    ax[2,1].set_ylabel('Diff-Close');

    st.pyplot (fig)


# Above are a series of plots that show the relationship between different differential price measurements and the differential close. In this study, the differece relates to the difference between a value at time t and the previous day value at time t-1. The Differential high, differential low, differential high-low and differential open-close appear to have a linear relationship with the differential close. However, only the differential open-close would be useful in an analysis. This because on a given day (time t), we can not know what the highs or lows are before hand till the day ends. However, we know the open value at the start of the trading period.

# Let's separate the data features and target variables. We will use Ridge regression in this case to make our model more generalizable

# In[123]:




# In[124]:


    X_reg = df_updated[['Open-Close']]
    y_reg = df_updated['Diff_Close']


# In[125]:


    X_reg = X_reg.loc[1:,:]
    y_reg = y_reg.iloc[1:]


# In[126]:


    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=0)


# We will perform a grid search and cross validation to determine optimal paramters for our regresison model

# In[127]:


    ridge = Ridge()
    alphas = [1e-15,1e-8,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,0,1,5,10,20,30,40,45,50,55,100]
    params = {'alpha': alphas}


# In[129]:


    ridge_regressor = GridSearchCV(ridge,params, scoring='neg_mean_squared_error',cv=10)
    ridge_regressor.fit(X_reg,y_reg)


# In[130]:


    st.text(ridge_regressor.best_score_)
    st.text(ridge_regressor.best_params_)


# Finally, let's produce a plot and see how it fits

# In[131]:


    np.shape(X_test_reg)


# In[133]:


    regr = Ridge(alpha=1e-15)
    regr.fit(X_train_reg, y_train_reg)

    y_pred = regr.predict(X_test_reg)
    y_pred_train = regr.predict(X_train_reg)

    st.text(f'R^2 value for test set is {regr.score(X_test_reg,y_test_reg)}')
    st.text(f'Mean squared error is {mean_squared_error(y_test_reg,y_pred)}')


    plt.scatter(df_updated['Open-Close'][1:],df_updated['Diff_Close'][1:],c='k')
    plt.plot(df_updated['Open-Close'][1:], (regr.coef_[0] * df_updated['Open-Close'][1:] + regr.intercept_), c='r' );
    plt.xlabel('Open-Close')
    plt.ylabel('Diff-Close')


# We obtained a mean square error of 0.58 which is fairly moderate. Our R^2 value basically says 54% of the variance in the
# differential close price is explained by the differential open-close price. Not so bad so far. But to be truly effective, we need to make use of statistics. Specifically, let's define a confidence interval around our predictions i.e prediction intervals.
#
# Prediction intervals give you a range for the prediction that accounts for any threshold of modeling error. Prediction intervals are most commonly used when making predictions or forecasts with a regression model, where a quantity is being predicted. We select the 95% confidence interval in this example such that our actual predictions fall into this range 99% of the time. For an in-depth overview and explanation please explore [machinelearningmastery](https://machinelearningmastery.com/prediction-intervals-for-machine-learning/)

# In[135]:


# In[136]:


    lower, upper, interval = predict_range(X_reg, y_reg,regr)


# In[138]:


    plt.scatter(X_reg ,df_updated['Diff_Close'][1:],c='k')
    plt.plot(X_reg , lower, c='b' )
    plt.plot(X_reg , (regr.coef_[0] * df_updated['Open-Close'][1:] + regr.intercept_), c='r' )
    plt.plot(X_reg , upper, c='g' )

#plt.errorbar(X_reg , (regr.coef_[0] * df_updated['Open-Close'][1:] + regr.intercept_),yerr=interval)
#

    plt.xlabel('Open-Close')
    plt.ylabel('Diff-Close')
    plt.legend(['Upper bound','Model','Lower bound']);
    st.pyplot (plt , use_container_width=True)


# Our prediction error corresponds to a value of **$1.82** in this example. Of course the parameters used to obtain our regression model (slope and intercept) also have confidence interval which we could calculate, but its the same process as highlighted above. From this plot, we can see that even with a 99% confidence interval, some values still fall outside the range. This highlights just how difficult it is forecasting stock price movements. So ideally we would suppliment our analysis with news, technical indicators and other parameters. In addition, our model is still quite limited. We can only make predictions about closing price on the same day i.e at time **t**. Even with the large uncertainty, this could prove useful in, for instance, options tradings.

# ## What does it all mean
#
# The stock screening strategy introduced could be a valuable tool for finding good stocks and minimizing loss. Certainly, in future projects, I could perform sentiment analysis myself in order to get information on every stock from the biggest movers list. There is also some room for improvement in the LSTM algorithm. If there was a way to minimize the bias and eliminate the implicit autocorrelation in the results, I’d love to hear about it. So far, the majority of research on the topic haven’t shown how to bypass this issue. Of course, more model tuning and data may help so any experts out there, please give me some feedback. The regression analysis seems pretty basic but could be a good way of adding extra information in our trading decisions. Of course, we could have gone with a more sophisticated approach; the differential prices almost appear to form an ellipse around the origin. Using a radial SVM classifier could also be a potential way of estimating stock movements.

# In[ ]:





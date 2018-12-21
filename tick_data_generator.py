

'''
Data preprocessing file for training data. 
Takes the level 1 tick data for a day.
Data file consists of only one instrument.
'''

# Tick Data courtesy nyxdata ftp://ftp.nyxdata.com/Historical%20Data%20Samples/Daily%20TAQ%20Sample%202017/

print("Tick data provided by nyxdata")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

pd.options.mode.chained_assignment = None

def genTradingDay(filename = 'AAPL_20180117.gz'):
    print(filename)
    file_path = './data/' + filename
    print(file_path)
    tick_df = pd.read_csv(file_path, compression='gzip', header=0, sep=',', quotechar='"', index_col=0)
    tick_df.index = pd.to_datetime(tick_df.index)

    # Only take bid prices from exchange 'Q'. Other bid prices seem to be unreliable
    to_be_filtered = (tick_df.Exchange == 'Q').values + (np.isfinite(tick_df['Trade Price'])).values

    tick_df = tick_df[to_be_filtered]

    tick_df['Bid_Price'] = tick_df['Bid_Price'].fillna(method = 'ffill')

    # trade dataframe
    df_trades = tick_df[np.isfinite(tick_df['Trade Price'])]

    # standardize the bid_prices
    mu_bid = df_trades['Bid_Price'].mean()
    std_bid = df_trades['Bid_Price'].std()
    df_trades['Bid_Price'] = (df_trades['Bid_Price'] - mu_bid)/std_bid
    # remove bid prices outliers
    df_trades = df_trades[np.abs(df_trades['Bid_Price'] - df_trades['Bid_Price'].rolling(25).mean()) <= (2 * df_trades['Bid_Price'].std())]

    # clean the trades
    df_trades = df_trades[np.abs(df_trades['Trade Price'] - df_trades['Trade Price'].rolling(25).mean()) <= (3 * df_trades['Trade Price'].rolling(25).std())]
    df_trades['Trade Price'] = (df_trades['Trade Price'] - mu_bid)/std_bid
    df_trading_day = df_trades['01-17-18 09:30:00': '01-17-18 15:55:00']
    # drop unnecessary columns
    df_trading_day = df_trading_day.drop(['Offer_Price', 'Offer_Size', 'Quote_Condition', 'Bid_Size', 'Exchange', 'Symbol', 'Sale Condition'], axis = 1)
    #print(df_trading_day.columns)

    time = df_trading_day.index - pd.Timedelta(hours = 9, minutes = 30)
    time_in_minutes = time.hour*60 + time.minute
    time_in_minutes = time_in_minutes.values
    mu_time = time_in_minutes.mean()
    std_time = time_in_minutes.std()
    df_trading_day['time_in_minutes'] = time_in_minutes

    momentum = (df_trading_day['Trade Price'].pct_change() * df_trading_day['Trade Volume']).rolling(window = 10).sum().fillna(0)
    df_trading_day['momentum'] = momentum

    """ print("")
    print("There are " + str(df_trading_day.shape[0]) + " number of ticks")
    print("")
    # a total of 210360 ticks
    """
    upperBound = (df_trading_day.index >= '01-17-18 15:54:00').argmax()

    return df_trading_day, upperBound, mu_bid, mu_time, std_bid, std_time

# This forms our training data. Each episode is of 1 minute in length.

# Randomly generate an episode of the data frame.

def genEpisode(df_trading_day, upperBound):

    # Start Index upper bounded by the index before the last 1 min
    start_index = np.random.randint(11, upperBound)
    start_time = df_trading_day.index[start_index]
    end_time = start_time + pd.Timedelta(minutes = 1)
    episodeSlice = df_trading_day[start_time:end_time]
    # end_index = episodeSlice.shape[0] + start_index
    return episodeSlice

# print(genEpisode(df_trading_day, upperBound).shape)

# partition the trade day data into 1 minute slices and return these slices as episodes
def partitionTradeDay(df_trading_day):
    dflist = []
    for group in df_trading_day.groupby(df_trading_day['time_in_minutes']):
        dflist.append(group[1])

    # exclude the last minute
    dflist = dflist[:-1]
    return dflist

df_trading_day, upperBound, mu_bid, mu_time, std_bid, std_time = genTradingDay()


""" 
# Explore for the number of ticks in a minute and a histogram of volumes every minute

minute_volumes = []
minute_ticks = []
minute_time = []
# generate 1000 different minute slices and plot histograms for volumes and trades
for _ in range(1000):
    thisSlice = genEpisode(df_trading_day, upperBound)
    time_from_open = thisSlice.index[0] - pd.Timedelta(hours = 9, minutes = 30)
    time_in_minutes = time_from_open.hour*60 + time_from_open.minute
    volume = thisSlice['Trade Volume'].sum()
    ticks = thisSlice.shape[0]
    minute_volumes.append(volume)
    minute_ticks.append(ticks)
    minute_time.append(time_in_minutes)

minute_ticks = np.array(minute_ticks)
minute_volumes = np.array(minute_volumes)
minute_time = np.array(minute_time)

minute_time = (minute_time - mu_time)/std_time

plt.title('Number of ticks in a minute')
plt.hist(minute_ticks)
plt.xlabel('# ticks')
plt.savefig('ticksHist.png')
plt.close()

plt.title('Volume in a minute')
plt.hist(minute_volumes)
plt.xlabel('Volume')
plt.savefig('volumeHist.png')
plt.close()

plt.title('Time Distribution of episodes')
plt.hist(minute_time)
plt.xlabel('Time (in minutes)')
plt.savefig('timeHist.png')
plt.close()
 """

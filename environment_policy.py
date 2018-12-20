# ---------------------------------RL agent ---------------------------------------
# State is defined as -- time_remaining(in secs), quantity remaining(in lotSizes), current_min, "Momentum", bid_price
# Action is an Integer <= quantity remaining
# Action taken specifies the number of lot sizes the agent decides to sell at Bid Price
# Orders are assumed to be immediate

# The problem to optimize for the RL agent is --
# Sell 2000 shares in lots at bid price of a given tick
# with minimum lot size 100 and maximum lot size 400
# Therefore, action space is discrete of size 5 (0, 1, 2, 3 , 4)

# "Momentum" for our purposes is the sum of product of last 10 returns and their volumes

# Policy gradient method is used for optimization
# The policy network has input as momentum(un-standardized), bid_price(standardized), quantity_remaining(in lotSizes)/time_remaining(in secs), current bar_time(standardized minutes)

import tensorflow as tf
import numpy as np
import pandas as pd
from dataGen import genEpisode, df_trading_day, upperBound, mu_time, std_time

class Policy(object):

    def __init__(self, obsSize, actSize, sess, optimizer, devicename = '/cpu:0'):
        # obsSize: size of your states
        # actSize: size of your action space

        # Input to the network is a vector of state representation
        # And the output of the network is the probability distribution over available actions
        with tf.device(devicename):
            n1 = 128
            n2 = 128
            n3 = 128
            state = tf.placeholder(tf.float32, [None, obsSize])
            l1 = tf.layers.dense(inputs = state, units = n1, activation = tf.nn.relu)
            l1 = tf.layers.dropout(inputs = l1, rate = 0.4)
            l2 = tf.layers.dense(inputs = l1, units = n2, activation = tf.nn.relu)
            l2 = tf.layers.dropout(inputs = l2, rate = 0.2)
            l3 = tf.layers.dense(inputs = l2, units = n3)
            l3 = tf.layers.dropout(inputs = l3, rate = 0.1)
            l4 = tf.layers.dense(inputs = l3, units = actSize)

            prob = tf.nn.softmax(l4)

            Q_estimate = tf.placeholder(tf.float32, [None])
            actions = tf.placeholder(tf.int32, [None])
            indices = tf.one_hot(actions, depth = actSize, dtype = tf.float32)

            prob_pred = tf.reduce_sum(tf.multiply(prob, indices), axis = 1)

            surrogate_loss = -1*tf.reduce_mean(tf.multiply(Q_estimate, tf.log(prob_pred)))

            self.train_op = optimizer.minimize(surrogate_loss)

            self.state = state
            self.prob = prob
            self.actions = actions
            self.Q_estimate = Q_estimate
            self.loss = surrogate_loss
            self.optimizer = optimizer
            self.sess = sess

    def compute_prob(self, states):
        return self.sess.run(self.prob, feed_dict = {self.state:states})

    def train(self, states, actions, Qs):
        self.sess.run(self.train_op, feed_dict = {self.state:states, self.actions:actions, self.Q_estimate:Qs})



class Environment(object):

    def __init__(self):
        self.episodeSlice = False
        self.done = False
        self.state = False
        self.endTime = False
        self.quantity_remaining = False
        self.time_remaining = False
        self.mu_time = mu_time
        self.std_time = std_time
        self.barTime = False
        self.vwap = False
        self.obsVwap = False
        self.index = False
        self.obsSize = 4
        self.actSize = 5
        self.action = False
        self.averagePrice = False

    def reset(self, qty = 20, epSlice = pd.Series([])):
        self.done = False
        # randomly generate an episode if no episode slice given (during training)
        if epSlice.empty:
            self.episodeSlice = genEpisode(df_trading_day, upperBound)
        else:
            self.episodeSlice = epSlice
        self.vwap = (self.episodeSlice['Trade Price']*self.episodeSlice['Trade Volume']).sum()/self.episodeSlice['Trade Volume'].sum()
        # The first tick of episode slice
        tick = self.episodeSlice.iloc[0]
        self.obsVwap = tick['Bid_Price']
        current_time = self.episodeSlice.index[0]
        self.index = 0
        current_time_in_minutes = current_time - pd.Timedelta(hours = 9, minutes = 30)
        self.endTime = current_time_in_minutes + pd.Timedelta(seconds = 60)
        self.time_remaining = (self.endTime - current_time_in_minutes).total_seconds()
        self.total_quantity = float(qty)   # total of 20 lot sizes by default
        if self.total_quantity == 0:
            self.done = True
        self.quantity_remaining = self.total_quantity
        self.averagePrice = 0.0
        self.barTime = (self.episodeSlice.iloc[0]['time_in_minutes'] - mu_time)/std_time
        # state for any given episode is the array of [momentum, bid_price, quantity_remaining/time_remaining, barTime]
        self.state = np.array([tick['momentum'], tick['Bid_Price'], self.quantity_remaining/self.time_remaining, self.barTime])
        return self.state, self.done

    # This is the function which executes one step of the action taken by the agent in a given episode
    # Input to the function is the time index where the action was taken and the action taken
    # The function outputs next state, reward and a boolean indicating whether the episode is complete
    def step(self, action):
        self.index += 1
        time_index = self.episodeSlice.index[self.index]
        tick = self.episodeSlice.iloc[self.index]
        self.action = min(action, self.quantity_remaining)
        self.quantity_remaining = self.quantity_remaining - self.action
        if self.quantity_remaining == 0 or time_index == self.episodeSlice.index[-1]:
            self.done = True
        current_time_in_minutes = time_index - pd.Timedelta(hours = 9, minutes = 30)
        self.time_remaining = (self.endTime - current_time_in_minutes).total_seconds()
        # next state is the array of [momentum, bid_price, quantity_remaining/time_remaining, barTime]
        self.state = np.array([tick['momentum'], tick['Bid_Price'], self.quantity_remaining/self.time_remaining, self.barTime])
        # Reward is the difference between price achieved by the action and the observed vwap, weighted by the action weight
        action_weight = self.action/self.total_quantity
        self.obsVwap = (self.episodeSlice[:time_index]['Trade Price']*self.episodeSlice[:time_index]['Trade Volume']).sum()/self.episodeSlice[:time_index]['Trade Volume'].sum()
        self.averagePrice += tick['Bid_Price']*action_weight
        if not self.done:
            reward = (tick['Bid_Price'] - self.obsVwap)*action_weight
        else:
            reward = (tick['Bid_Price'] - self.obsVwap)*action_weight - self.quantity_remaining
        return self.state, reward, self.done, self.vwap

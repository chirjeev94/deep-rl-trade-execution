
import tensorflow as tf
import numpy as np
import pandas as pd
from environment import Policy, Environment
import time
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataGen import genEpisode, df_trading_day, upperBound, mu_time, std_time, mu_bid, std_bid
from environment import Policy, Environment


class ExecutionAgent:

    def __init__(self):
        ##self.Policy

    def train(self):
        pass



    def execute_minute(self):
        pass


    def execute_day(self,dataFileName = 'AAPL_20180117.gz',lotsToSell = 2500):
        pass

        '''
        # This implements the 1-min trade agent to trade for an entire day
        # Idea is to get build a Volume profile for the day
        # Use this volume profile to "bucketize" the trade volume througout the day's minute bars
        # Feed these buckets to the trade agent, and let the trade agent decide on the optimum trade strategy for every minute
        '''
        sns.set_style('whitegrid')
        # saved file with percentage volume targets
        file_name = 'targetPercentage'
        infile = open(file_name, 'rb')

        targetPct = np.load(infile)
        # print(targetPct)
        tickr = dataFileName.split('_')[0]
    

        orderTargets = lotsToSell*targetPct
        orderTargetsShift = np.roll(orderTargets, 1)
        orderTargetsShift[0] = 0

        orders = orderTargets - orderTargetsShift

        orders = np.rint(orders)
        #print(len(orders))

        diff = np.sum(orders) - lotsToSell

        # subtract the difference from last order to make to total sum equal to lotSizes
        orders[-1] = orders[-1] - diff

        #print(np.sum(orders))

        episodeList = partitionTradeDay(df_trading_day)

        #plt.plot(df_trading_day.index, (df_trading_day['Bid_Price']*std_bid + mu_bid), label = 'Bid Prices', c = 'b')
        #plt.show()

        alpha = 1e-3
        optimizer_p = tf.train.AdamOptimizer(alpha)
        env = Environment()
        obsSize = env.obsSize
        actSize = env.actSize

        sess = tf.Session()
        actor = Policy(obsSize, actSize, sess, optimizer_p)

        saver = tf.train.Saver()
        saver.restore(sess, './model/parameters.ckpt')
        print('Model Restored')

        ACTS = []
        traded_price = []
        traded_time = []
        shortfall_list = []

        # apply the trade agent to each episode in the list
        for i in range(len(episodeList)):
            # Number of orders to be executed for the particular episode
            order = orders[i]
            episode = episodeList[i]
            # reset the environment for the particular slice and order qty
            obs, done = env.reset(qty=order, epSlice=episode)
            acts = [0]
            grad = False
            while not done:
                prob = actor.compute_prob(np.expand_dims(obs, 0))
                if grad:
                    action_prob = np.exp(prob[0]**2/prev_prob)
                    action_prob = action_prob/np.sum(action_prob)
                    prev_prob = prob[0]
                else:
                    grad = True
                    prev_prob = prob[0]
                    action_prob = prob[0]
                #action = np.random.choice(actSize, p = action_prob.flatten())
                action = action_prob.argmax()
                newobs, _, done, _ = env.step(action)
                obs = newobs
                acts.append(env.action)
                if env.action > 0.5:
                    traded_price.append(newobs[1])
                    traded_time.append(env.episodeSlice.index[env.index])
            
            pad = [0 for j in range(env.episodeSlice.shape[0] - len(ACTS))]
            acts += pad
            sharesSold = sum(acts)
            # Weight of this particular episode in our overall selling process
            weight = sharesSold/float(lotsToSell)
            # if there is a shortfall add it to the next order (primitive should think of something else)
            shortfall = order - sharesSold
            shortfall_list.append(shortfall)
            if i < len(episodeList) - 1:
                orders[i + 1] += shortfall
            ACTS += acts
            

        traded_price = np.array(traded_price)
        traded_price = traded_price*std_bid + mu_bid
        ACTS = np.array(ACTS)
        agent_ap = np.sum(ACTS[ACTS > 0]*traded_price)/np.sum(ACTS)
        vwap = (df_trading_day['Trade Price']*df_trading_day['Trade Volume']).sum()/np.sum(df_trading_day['Trade Volume'])*std_bid + mu_bid

        print("VWAP during the day is " + str(vwap))
        print("Agent realized average price is " + str(agent_ap))
        print("Total lots to be sold " + str(lotsToSell))
        print("Number of lots sold " + str(np.sum(ACTS) + 1))
        print("Total number of trades " + str(np.sum(ACTS > 0)))

        """ 
        plt.plot(df_trading_day.index, (df_trading_day['Trade Price']*std_bid + mu_bid), label = tickr, c = 'b')
        plt.plot(traded_time, traded_price, label = 'Agent trade points', c = 'y')
        plt.axhline(y = agent_ap, label = 'Agent AP', color = 'g')
        plt.axhline(y = vwap, label = 'VWAP', color = 'r', linestyle = 'dashed')
        plt.legend()
        plt.savefig(tickr + '.png')
        plt.close()
        plt.plot(df_trading_day.index, df_trading_day['Trade Volume'].cumsum()/df_trading_day['Trade Volume'].sum(), label = 'ActualVolumePct')
        plt.plot(traded_time, traded_price.cumsum()/traded_price.sum(), label = 'AgentVolumePct')
        plt.legend()
        plt.savefig(tickr + '_volume.png')
        plt.close()
        """





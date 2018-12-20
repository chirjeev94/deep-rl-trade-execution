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

sns.set_style('whitegrid')


class ExecutionAgent:

    def __init__(self, alpha):
        self.optimizer = tf.train.AdamOptimizer(alpha)
        self.environment = Environment()
        ##self.Policy

    def train(self):

        pass

    def execute_minute(self):
        
        env = self.environment
        obsSize = env.obsSize
        actSize = env.actSize

        sess = tf.Session()
        actor = Policy(obsSize, actSize, sess, self.optimizer)

        saver = tf.train.Saver()
        saver.restore(sess, './model/parameters.ckpt')
        print('Model Restored')

        adv = []

    def execute_day(self,dataFileName = 'AAPL_20180117.gz',lotsToSell = 2500):
        
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


        iterations = 1000
        agent_ap = []
        vwap_episode = []
        trade_times = []

        for ite in range(iterations):

            ACTS = [0]
            probs = [[0.0, 0.0, 0.0, 0.0, 0.0]]
            unnormalized_prob = [[0.0, 0.0, 0.0, 0.0, 0.0]]
            obs, done = self.environment.reset()
            grad = False
            while not done:
                prob = actor.compute_prob(np.expand_dims(obs, 0))
                if grad:
                    action_prob = np.exp(prob[0] ** 2 / prev_prob)
                    action_prob = action_prob / np.sum(action_prob)
                    prev_prob = prob[0]
                else:
                    grad = True
                    prev_prob = prob[0]
                    action_prob = prob[0]
                # action = np.random.choice(actSize, p = action_prob.flatten())
                action = np.argmax(action_prob)
                newobs, _, done, _ = env.step(action)
                obs = newobs
                ACTS.append(env.action)
                probs.append(action_prob.tolist())
                unnormalized_prob.append(prob[0].tolist())
                if env.action > 0.5:
                    trade_times.append(env.episodeSlice.index[env.index])

            pad = [0 for i in range(env.episodeSlice.shape[0] - len(ACTS))]
            prob_pad = [[0.0, 0.0, 0.0, 0.0, 0.0] for i in range(len(pad))]
            probs += prob_pad
            unnormalized_prob += prob_pad
            ACTS += pad
            adv.append(env.averagePrice - env.vwap)
            agent_ap.append(env.averagePrice * std_bid + mu_bid)
            vwap_episode.append(env.vwap * std_bid + mu_bid)
            probs = np.array(probs).reshape(-1, 5)
            unnormalized_prob = np.array(unnormalized_prob).reshape(-1, 5)

        agent_ap = np.array(agent_ap)
        vwap_episode = np.array(vwap_episode)
        """ plt.title('Agent performance relative to VWAP')
        plt.hist((agent_ap - vwap_episode)/ agent_ap)
        plt.xlabel('difference between agent price and VWAP')
        plt.savefig('agentPerformance.png')
        plt.close() """

        print(np.mean((agent_ap - vwap_episode) / agent_ap))
        print(np.std((agent_ap - vwap_episode) / agent_ap))
        """ 
        unstandardized_prices = env.episodeSlice['Trade Price']*std_bid + mu_bid
        
        plt.plot(env.episodeSlice.index, unstandardized_prices, label = 'Trade Prices', color = 'b')
        plt.axhline(y = agent_ap[0], label = 'Agent AP', color = 'g')
        plt.axhline(y = vwap_episode[0], label = 'VWAP', color = 'r')
        for t in trade_times:
            plt.axvline(x = t, color = 'black', marker = '|')
        plt.legend()
        plt.savefig('tradeStats.png')
        plt.close()
        
        
        
        fig, ax = plt.subplots(5, 1, sharex = True)
        plt.title('Acion Probabilities')
        ax[0].plot(env.episodeSlice.index, probs[:, 0], label = 'Prob_0')
        ax[0].legend()
        ax[1].plot(env.episodeSlice.index, probs[:, 1], label = 'Prob_1')
        ax[1].legend()
        ax[2].plot(env.episodeSlice.index, probs[:, 2], label = 'Prob_2')
        ax[2].legend()
        ax[3].plot(env.episodeSlice.index, probs[:, 3], label = 'Prob_3')
        ax[3].legend()
        ax[4].plot(env.episodeSlice.index, probs[:, 4], label = 'Prob_4')
        ax[4].legend()
        plt.savefig('Probability.png')
        plt.close()
        
        fig, ax = plt.subplots(5, 1, sharex = True)
        plt.title('Output Probabilities')
        ax[0].plot(env.episodeSlice.index, unnormalized_prob[:, 0], label = 'Prob_0')
        ax[0].legend()
        ax[1].plot(env.episodeSlice.index, unnormalized_prob[:, 1], label = 'Prob_1')
        ax[1].legend()
        ax[2].plot(env.episodeSlice.index, unnormalized_prob[:, 2], label = 'Prob_2')
        ax[2].legend()
        ax[3].plot(env.episodeSlice.index, unnormalized_prob[:, 3], label = 'Prob_3')
        ax[3].legend()
        ax[4].plot(env.episodeSlice.index, unnormalized_prob[:, 4], label = 'Prob_4')
        ax[4].legend()
        plt.savefig('outputProbability.png')
        plt.close()
        #print(agent_ap - vwap_episode)
        #print(ACTS)
        #print(len(ACTS))
        #print(env.episodeSlice.shape[0])
         """

    def execute_day(self):
        pass

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

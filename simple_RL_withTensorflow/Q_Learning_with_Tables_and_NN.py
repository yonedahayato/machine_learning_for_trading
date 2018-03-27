import gym
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import tensorflow as tf

from trading_env import Trading_Env
from helper.timer import Timer

class LeinforceRearning():
    def __init__(self, game_name="FrozenLake"):
        if game_name == "FrozenLake":
            self.train_env = gym.make("FrozenLake-v0")
            self.test_env = gym.make("FrozenLake-v0")

        elif game_name == "Trading":
            self.train_env = Trading_Env(train=True)
            self.test_env = Trading_Env(train=False)
        else:
            raise("It is invalid game name.")

        #create lists to contain total rewards and steps per episode
        self.jList = []
        self.rList = []

        self.s = None
        self.a = None
        self.s1 = None
        self.r = None
        self.check_actions = []
        self.check_statuses = []

        self.d = None

    def set_parameters(self, **params):
        self.num_episodes = params["num_episodes"]

    def set_learning_parameters(self):
        self.lr = 0.8
        self.y = 0.95
        self.num_episodes = 2000
        # self.num_episodes = 10
        self.step_num = 200

    def initialize_Qtable_with_zeros(self):
        Q = np.zeros([self.train_env.observation_space.n, self.train_env.action_space.n])

        self.Q = Q
        return Q

    def status_check(self, episode=None, step=None, Qtable=False):
        print("="*5 + "episode:{}, step:{}".format(episode, step) + "="*5)
        if Qtable:
            print("Q-table")
            print(self.Q)

        print("state:{}".format(self.s))
        print("action:{}".format(self.a))
        print("state1:{}".format(self.s1))
        print("reward:{}".format(self.r))

    def chose_action_by_greedily_picking_from_Qtable(self, episode, train=True):
        #Choose an action by greedily (with noise) picking from Q table
        if train:
            a = np.argmax(self.Q[self.s,:] + np.random.randn(1,self.train_env.action_space.n)*(1./(episode+1)))
        else:
            a = np.argmax(self.Q[self.s,:] + np.random.randn(1,self.test_env.action_space.n)*(1./(episode+1)))

        self.a = a
        return a

    def get_new_state_reward_from_environment(self, action, train=True):
        #Get new state and reward from environment
        if train:
            s1, r, d, _ = self.train_env.step(action)
        else:
            s1, r, d, _ = self.test_env.step(action)

        self.s1 = s1
        self.r = r
        self.d = d
        return s1, r, d

    def update_Qtable_with_new_knowledge(self, s, a, r, s1):
        #Update Q-Table with new knowledge
        Q = self.Q.copy()

        lr = self.lr
        y = self.y

        Q[s, a] = Q[s, a] + lr*(r + y * np.max(Q[s1,:]) - Q[s,a])

        self.Q = Q.copy()
        return Q

    def result(self, rList, Qtable=False, check=False, train=True):
        print("Score over time: " + str(sum(rList)/self.num_episodes))
        print("Step num over time: " + str(sum(self.jList)/self.num_episodes))
        if Qtable:
            print("Final Q-Table Values")
            print(self.Q)

        if check:
            if train:
                env = self.train_env
            else:
                env = self.test_env

            print("action_space.n")
            print(env.action_space.n)
            print("actions")
            print(check_actions)

            print("observation_space.n")
            print(env.observation_space.n)
            print("statuses")
            print(check_statuses)

    def train(self):
        Q = self.initialize_Qtable_with_zeros()
        self.set_learning_parameters()

        timer = Timer()
        timer.start(name="train_all")

        for i in range(self.num_episodes):
            timer.start(name="train_episode_{}".format(i))

            #Reset environment and get first new observation
            s = self.train_env.reset() # start
            self.s = s

            rAll = 0
            d = False
            j = 0
            actions = []
            statuses = [s]

            #The Q-Table learning algorithm
            while j < self.step_num: # step
                j+=1

                a = self.chose_action_by_greedily_picking_from_Qtable(episode=i, train=True)
                s1, r, d = self.get_new_state_reward_from_environment(action = a, train=True)
                Q = self.update_Qtable_with_new_knowledge(s, a, r, s1)

                self.status_check(episode=i+1, step=j, Qtable=False)

                actions.append(a)
                statuses.append(s1)
                rAll += r
                self.s = s1
                s = s1
                if d == True:
                    break

            self.jList.append(j)
            self.rList.append(rAll)
            self.check_actions.append(actions)
            self.check_statuses.append(statuses)

            timer.stop(name="train_episode_{}".format(i))

        self.result(self.rList, Qtable=False, check=False, train=True)

        timer.stop(name="train_all")
        timer.result_write_csv()

    def test(self):
        Q = self.Q
        timer = Timer()
        timer.start(name="test")

        s = self.test_env.reset() # start
        self.s = s

        rAll = 0
        d = False
        j = 0

        while j < self.step_num: # step
            j+=1

            a = self.chose_action_by_greedily_picking_from_Qtable(episode=self.num_episodes, train=False)
            s1, r, d = self.get_new_state_reward_from_environment(action = a, train=False)
            Q = self.update_Qtable_with_new_knowledge(s, a, r, s1)

            # self.status_check(episode="test", step=j, Qtable=False)

            rAll += r
            self.s = s1
            s = s1
            if d == True:
                break

        print("test resutl {}".format(rAll))

        timer.stop(name="test")
        timer.result_write_csv()

def main():
    # LR = LeinforceRearning(game_name="FrozenLake")
    LR = LeinforceRearning(game_name="Trading")
    LR.train()
    LR.test()

def main_tmp():

    env = gym.make("FrozenLake-v0")

    tf.reset_default_graph()

    #These lines establish the feed-forward part of the network used to choose actions
    action_num = 16
    inputs1 = tf.placeholder(shape=[1,action_num],dtype=tf.float32)
    W = tf.Variable(tf.random_uniform([16,4],0,0.01))
    Qout = tf.matmul(inputs1,W)
    predict = tf.argmax(Qout,1)

    #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
    nextQ = tf.placeholder(shape=[1,4],dtype=tf.float32)
    loss = tf.reduce_sum(tf.square(nextQ - Qout))
    trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    updateModel = trainer.minimize(loss)

    # init = tf.initialize_all_variables()
    init = tf.global_variables_initializer()

    # Set learning parameters
    y = .99
    e = 0.1
    num_episodes = 2000
    #create lists to contain total rewards and steps per episode
    jList = []
    rList = []
    with tf.Session() as sess:
        sess.run(init)
        for i in range(num_episodes):
            #Reset environment and get first new observation
            s = env.reset()
            rAll = 0
            d = False
            j = 0
            #The Q-Network
            while j < 99:
                j+=1
                #Choose an action by greedily (with e chance of random action) from the Q-network
                a, allQ = sess.run([predict,Qout],feed_dict={inputs1:np.identity(16)[s:s+1]})
                if np.random.rand(1) < e:
                    a[0] = env.action_space.sample()

                #Get new state and reward from environment
                s1,r,d,_ = env.step(a[0])
                #Obtain the Q' values by feeding the new state through our network
                Q1 = sess.run(Qout,feed_dict={inputs1:np.identity(16)[s1:s1+1]})
                #Obtain maxQ' and set our target value for chosen action.
                maxQ1 = np.max(Q1)
                targetQ = allQ
                targetQ[0,a[0]] = r + y*maxQ1
                #Train our network using target and predicted Q values
                _,W1 = sess.run([updateModel,W],feed_dict={inputs1:np.identity(16)[s:s+1],nextQ:targetQ})
                rAll += r
                s = s1
                if d == True:
                    #Reduce chance of random action as we train the model.
                    e = 1./((i/50) + 10)
                    break
            jList.append(j)
            rList.append(rAll)
    print("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")
    plt.plot(rList)

if __name__ == "__main__":
    main()
    # main_tmp()

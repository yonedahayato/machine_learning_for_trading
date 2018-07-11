import gym
import matplotlib as mpl
mpl.use('TkAgg')
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import tensorflow as tf

from helper.make_graph import Make_Graph
from helper.params_save_load import *
from helper.timer import Timer
from trading_env import Trading_Env

class ReinforceLearning():
    def __init__(self, game_name="FrozenLake", status_value=False):
        self.game_name = game_name
        if game_name == "FrozenLake":
            self.train_env = gym.make("FrozenLake-v0")
            self.test_env = gym.make("FrozenLake-v0")

        elif game_name == "Trading":
            self.train_env = Trading_Env(train=True, status_value=status_value)
            self.test_env = Trading_Env(train=False, status_value=status_value)
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

        self.set_learning_parameters()

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

    def reset_environment_and_get_first_new_observation(self, train=True):
        if train:
                s = self.train_env.reset()
        else:
                s = self.test_env.reset()
        self.s = s

        rAll = 0
        d = False
        j = 0
        actions = []
        statuses = [s]

        return rAll, d, j, actions, statuses, s

    def choose_action_by_greedily_picking_from_Qtable(self, episode, train=True):
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
            if self.game_name == "FrozenLake":
                s1, r, d, _ = self.train_env.step(action)
            elif self.game_name == "Trading":
                s1, r, d, _ = self.train_env.step(action)
        else:
            if self.game_name == "FrozenLake":
                s1, r, d, _ = self.test_env.step(action)
            elif self.game_name == "Trading":
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

    def result(self, Qtable=False, check=False, train=True):
        print("="*5 + " resutl " + "="*5)
        print("Score over time: " + str(sum(self.rList)/self.num_episodes))
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
        train_f = True
        Q = self.initialize_Qtable_with_zeros()

        mg_train_reward = Make_Graph(file_name="train_reward", Id_name="episode", value_name="reward")
        timer = Timer(file_name="train")
        timer.start(name="train_all")

        for i in range(self.num_episodes):
            timer.start(name="train_episode_{}".format(i))

            rAll, d, j, actions, statuses, s = self.reset_environment_and_get_first_new_observation(train=train_f)

            #The Q-Table learning algorithm
            while j < self.step_num: # step
                j+=1

                a = self.choose_action_by_greedily_picking_from_Qtable(episode=i, train=train_f)
                s1, r, d = self.get_new_state_reward_from_environment(action = a, train=train_f)
                Q = self.update_Qtable_with_new_knowledge(s, a, r, s1)

                self.status_check(episode=i+1, step=j, Qtable=False)

                actions.append(a)
                statuses.append(s1)
                rAll += r
                self.s, s = s1, s1
                if d == True:
                    break

            self.jList.append(j)
            self.rList.append(rAll)
            self.check_actions.append(actions)
            self.check_statuses.append(statuses)

            timer.stop(name="train_episode_{}".format(i))
            mg_train_reward.data_input(Id=i, value=rAll)

            if self.game_name == "Trading":
                if rAll >= max(self.rList):
                    best_reward_trading_stock_list = self.train_env.observation_space.stock_data_list
                    best_episode = i

        self.result(Qtable=False, check=False, train=train_f)

        timer.stop(name="train_all")
        timer.result_write_csv()
        mg_train_reward.save_line_graph()

        if self.game_name == "Trading":
            for cnt, stock_data_df in enumerate(best_reward_trading_stock_list):
                mg_chart_graph = Make_Graph(file_name="best_reward_chart_graph_stock_train:{}_episode:{}".format(cnt, best_episode), \
                                    Id_name="date", value_name="close")
                mg_chart_graph.save_chart_graph(stock_data_df)

    def test(self):
        train_f = False

        Q = self.Q
        timer = Timer(file_name="test")
        timer.start(name="test")

        rAll, d, j, actions, statuses, s = self.reset_environment_and_get_first_new_observation(train=train_f)

        while j < self.step_num: # step
            j+=1

            a = self.choose_action_by_greedily_picking_from_Qtable(episode=self.num_episodes, train=train_f)
            s1, r, d = self.get_new_state_reward_from_environment(action = a, train=train_f)
            Q = self.update_Qtable_with_new_knowledge(s, a, r, s1)

            self.status_check(episode="test", step=j, Qtable=False)

            rAll += r
            self.s = s1
            s = s1
            if d == True:
                break

        print("="*10)
        print("test resutl {}".format(rAll))

        timer.stop(name="test")
        timer.result_write_csv()

        if self.game_name == "Trading":
            for cnt, stock_data_df in enumerate(self.test_env.observation_space.stock_data_list):
                mg_chart_graph = Make_Graph(file_name="best_reward_chart_graph_stock_test:{}".format(cnt), \
                                    Id_name="date", value_name="close")
                mg_chart_graph.save_chart_graph(stock_data_df)

class ReinforceLearning_NN(ReinforceLearning):
    def __init__(self, game_name="FrozenLake", status_value=False):
        ReinforceLearning.__init__(self, game_name=game_name, status_value=status_value)
        self.status_value = status_value

        self.make_network_with_tensorflow()
        self.params_save_path = ""

    def make_network_with_tensorflow(self):
        msg = "[make_network_with_tensorflow]: "
        tf.reset_default_graph()

        #These lines establish the feed-forward part of the network used to choose actions
        if self.game_name == "FrozenLake" or (self.game_name == "Trading" and not self.status_value):
            self.inputs1 = tf.placeholder(shape=[1, self.train_env.observation_space.n],dtype=tf.float32)
        elif self.game_name == "Trading" and self.status_value:
            # input [adjust close / SMA, Bollinger Band value, P/E ratio]
            self.inputs1 = tf.placeholder(shape=[1, self.train_env.observation_space.stock_num *
                                                    self.train_env.observation_space.max_status_num], dtype=tf.float32)
            self.input_shape = (1, self.train_env.observation_space.stock_num *
                                    self.train_env.observation_space.max_status_num)
        else:
            print(msg + "game name is invalid")

        if self.game_name == "FrozenLake" or (self.game_name == "Trading" and not self.status_value):
            self.W = tf.Variable(tf.random_uniform([self.train_env.observation_space.n, self.train_env.action_space.n],0,0.01))
        elif self.game_name == "Trading" and self.status_value:
            self.W = tf.Variable(tf.random_uniform([self.train_env.observation_space.stock_num * self.train_env.observation_space.max_status_num,
                                                    self.train_env.action_space.n], 0, 0.01))
        else:
            print(msg + "game name is invalid")

        self.Qout = tf.matmul(self.inputs1, self.W)
        self.predict = tf.argmax(self.Qout,1)

        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.nextQ = tf.placeholder(shape=[1,self.train_env.action_space.n],dtype=tf.float32)
        loss = tf.reduce_sum(tf.square(self.nextQ - self.Qout))

        self.trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        self.updateModel = self.trainer.minimize(loss)

        self.init = tf.global_variables_initializer()

    def set_learning_parameters(self):
        self.y = .99
        self.e = 0.1
        self.num_episodes = 4000
        # self.num_episodes = 10
        self.step_num = 200

    def adjust_status_no(self, s, network=""):
        if self.status_value:
            return

        if not self.game_name == "Trading":
            return s

        if network == "input":
            s += 1
        elif network == "output":
            s -= 1
        else:
            raise("[adjust status no]: invalid network's in/out")

        return s

    def choose_action_by_greedily_from_the_QNetwork(self, s, train=True):
        #Choose an action by greedily (with e chance of random action) from the Q-network
        if self.game_name == "FrozenLake" or (self.game_name == "Trading" and not self.status_value):
            s = self.adjust_status_no(s, network="input")
            a, allQ = self.sess.run([self.predict, self.Qout],feed_dict={self.inputs1 : np.identity(self.train_env.observation_space.n)[s:s+1]})
        elif self.game_name == "Trading" and self.status_value:
            a, allQ = self.sess.run([self.predict, self.Qout], feed_dict={self.inputs1 : np.reshape(np.array(s), self.input_shape)})

        if np.random.rand(1) < self.e:
            if train:
                a[0] = self.train_env.action_space.sample()
            else:
                a[0] = self.test_env.action_space.sample()

        self.a = a[0]

        return a, allQ

    def set_target_value_for_chosen_action(self, s1, allQ, a):
        msg = "[set_target_value_for_chosen_action]: "

        #Obtain the Q' values by feeding the new state through our network
        if self.game_name == "FrozenLake" or (self.game_name == "Trading" and not self.status_value):
            s1 = self.adjust_status_no(s1, network="input")
            Q1 = self.sess.run(self.Qout,feed_dict={self.inputs1:np.identity(self.train_env.observation_space.n)[s1:s1+1]})
        elif self.game_name == "Trading" and self.status_value:
            Q1 = self.sess.run(self.Qout, feed_dict={self.inputs1: np.reshape(np.array(s1), self.input_shape)})
        else:
            print(msg + "game name is invalid")

        #Obtain maxQ' and set our target value for chosen action.
        maxQ1 = np.max(Q1)
        targetQ = allQ
        targetQ[0, a[0]] = self.r + self.y * maxQ1
        self.targetQ = targetQ

        return targetQ

    def train_network_using_target_predicted_Q_values(self, s):
        msg = "[train_network_using_target_predicted_Q_values]: "

        if self.game_name == "FrozenLake" or (self.game_name == "Trading" and not self.status_value):
            s = self.adjust_status_no(s, network="input")
            _, W1 = self.sess.run([self.updateModel, self.W],
                feed_dict={self.inputs1: np.identity(self.train_env.observation_space.n)[s:s+1],self.nextQ:self.targetQ})
        elif self.game_name == "Trading" and self.status_value:
            _, W1 = self.sess.run([self.updateModel, self.W],
                    feed_dict={self.inputs1: np.reshape(np.array(s), self.input_shape), self.nextQ : self.targetQ})
        else:
            print(msg + "game name is invalid")

    def print_result(self):
        print("Percent of succesful episodes: " + str(sum(self.rList) / self.num_episodes) + "%")

    def reset_environment_and_calculate_state(self, train=True):
        msg = "[reset_environment_and_calculate_state]: "
        if train:
                s = self.train_env.reset()
        else:
                s = self.test_env.reset()

        self.s = s

        rAll = 0
        d = False
        j = 0
        actions = []
        statuses = [s]

        return rAll, d, j, actions, statuses, s

    def train(self):
        train_f = True

        mg_train_reward = Make_Graph(file_name="train_reward", Id_name="episode", value_name="reward")
        timer = Timer(file_name="train")
        timer.start(name="train_all")

        with tf.Session() as sess:
            self.sess = sess

            try:
                lp = Load_Params()
                lp.load(sess, self.params_save_path)
            except Exception as e:
                print(e)
                sess.run(self.init)

            for i in range(self.num_episodes):
                timer.start(name="train_episode_{}".format(i))

                #Reset environment and get first new observation
                # rAll, d, j, actions, statuses, s = self.reset_environment_and_get_first_new_observation(train=train_f)
                rAll, d, j, actions, statuses, s = self.reset_environment_and_calculate_state(train=train_f)

                #The Q-Network
                while j < self.step_num:
                    j+=1

                    a, allQ = self.choose_action_by_greedily_from_the_QNetwork(s, train=train_f)
                    s1, r, d = self.get_new_state_reward_from_environment(a[0], train=train_f)
                    if self.game_name == "FrozenLake" or (self.game_name == "Trading" and not self.status_value):
                        s1 = self.adjust_status_no(s1, network="output")

                    self.s1 = s1

                    targetQ = self.set_target_value_for_chosen_action(s1, allQ, a)
                    self.train_network_using_target_predicted_Q_values(s)

                    # self.status_check(episode=i+1, step=j, Qtable=False)
                    rAll += r
                    self.s, s = s1, s1

                    if d == True:
                        # Reduce chance of random action as we train the model.
                        self.e = 1./((i/50) + 10)
                        break

                self.jList.append(j)
                self.rList.append(rAll)

                timer.stop(name="train_episode_{}".format(i))
                mg_train_reward.data_input(Id=i, value=rAll)

                if self.game_name == "Trading":
                    if rAll >= max(self.rList):
                        best_reward_trading_stock_list = self.train_env.observation_space.stock_data_list
                        best_episode = i
                    else:
                        print("rAll: {}, max_rList: {}".format(rAll, max(self.rList)))

            sp = Save_Params()
            self.params_save_path = sp.save(self.sess, file_name="Q_Learning_{}.ckpt".format(self.game_name))

        self.print_result()

        timer.stop(name="train_all")
        timer.result_write_csv()
        mg_train_reward.save_line_graph()

        if self.game_name == "Trading":
            for cnt, stock_data_df in enumerate(best_reward_trading_stock_list):
                mg_chart_graph = Make_Graph(file_name="best_reward_chart_graph_stock_train:{}_episode:{}".format(cnt, best_episode), \
                                    Id_name="date", value_name="close")
                mg_chart_graph.save_path = mg_train_reward.save_path
                mg_chart_graph.save_chart_graph(stock_data_df)

    def test(self):
        train_f = False

        timer = Timer(file_name="test")
        timer.start(name="test")

        with tf.Session() as sess:
            self.sess = sess

            try:
                lp = Load_Params()
                lp.load(sess, self.params_save_path)
            except Exception as e:
                print(e)
                sess.run(self.init)

            # rAll, d, j, actions, statuses, s = self.reset_environment_and_get_first_new_observation(train=train_f)
            rAll, d, j, actions, statuses, s = self.reset_environment_and_calculate_state(train=train_f)

            while j < self.step_num: # step
                j+=1
                a, allQ = self.choose_action_by_greedily_from_the_QNetwork(s, train=train_f)
                s1, r, d = self.get_new_state_reward_from_environment(a[0], train=train_f)
                if self.game_name == "FrozenLake" or (self.game_name == "Trading" and not self.status_value):
                    s1 = self.adjust_status_no(s1, network="output")
                self.s1 = s1

                targetQ = self.set_target_value_for_chosen_action(s1, allQ, a)
                self.train_network_using_target_predicted_Q_values(s)

                # self.status_check(episode="test", step=j, Qtable=False)
                rAll += r
                self.s, s = s1, s1

                if d == True:
                    break

        print("test resutl {}".format(rAll))

        timer.stop(name="test")
        timer.result_write_csv()

        if self.game_name == "Trading":
            for cnt, stock_data_df in enumerate(self.test_env.observation_space.stock_data_list):
                mg_chart_graph = Make_Graph(file_name="best_reward_chart_graph_stock_test:{}".format(cnt), \
                                    Id_name="date", value_name="close")
                mg_chart_graph.save_chart_graph(stock_data_df)

def main():
    # RL = ReinforceLearning(game_name="FrozenLake")
    RL = ReinforceLearning(game_name="Trading")
    RL.train()
    RL.test()
    RL.result(Qtable=False, check=False, train=True)

def main_NN():
    # RL_NN = ReinforceLearning_NN(game_name="FrozenLake")
    RL_NN = ReinforceLearning_NN(game_name="Trading", status_value=True)
    for i in range(2):
        print("== {} ==".format(i))
        RL_NN.train()
        RL_NN.test()
        RL_NN.result(Qtable=False, check=False, train=True)

if __name__ == "__main__":
    # main()
    main_NN()

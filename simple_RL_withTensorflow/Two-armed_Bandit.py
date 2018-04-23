import numpy as np
import tensorflow as tf

class Bandits:
    def __init__(self, bandits = [0.2, 0, -0.2, -5]):
        # The Bandits
        self.bandits = bandits
        self.num_bandits = len(self.bandits)

    def set_bandits(self, bandits):
        self.bandits = bandits
        self.num_bandits = len(self.bandits)

    def pullBandit(self, action):
        bandit = self.bandits[action]

        #Get a random number.
        result = np.random.randn(1)
        if result > bandit:
            #return a positive reward.
            return 1
        else:
            #return a negative reward.
            return -1

class set_parameters:
    def __init__(self, num_bandits):
        self.total_episodes = 1000 #Set total number of episodes to train agent on.
        self.total_reward = np.zeros(num_bandits) #Set scoreboard for bandits to 0.
        self.e = 0.1 #Set the chance of taking a random action.

        self.learning_rate = 0.001

class Agent(set_parameters):
    def __init__(self, num_bandits):
        # The Agent
        self.num_bandits = num_bandits
        set_parameters.__init__(self, num_bandits)

        self.make_graph()

    def make_graph(self):
        tf.reset_default_graph()

        #These two lines established the feed-forward part of the network. This does the actual choosing.
        self.weights = tf.Variable(tf.ones([self.num_bandits]))
        self.chosen_action = tf.argmax(self.weights,0)

        #The next six lines establish the training proceedure. We feed the reward and chosen action into the network
        #to compute the loss, and use it to update the network.
        self.reward_holder = tf.placeholder(shape=[1],dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[1],dtype=tf.int32)
        self.responsible_weight = tf.slice(self.weights, self.action_holder, [1])
        loss = -(tf.log(self.responsible_weight) * self.reward_holder)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        self.update = optimizer.minimize(loss)

    def update_network(self, reward, action):
        _, resp, ww = sess.run([self.update, self.responsible_weight, self.weights],
                            feed_dict={self.reward_holder:[reward], self.action_holder:[action]})

        return resp, ww

class Two_Armed_Bandit(set_parameters):
    def __init__(self):
        self.Bandits = Bandits()
        self.Agent = Agent(self.Bandits.num_bandits)

        set_parameters.__init__(self, self.Bandits.num_bandits)

    def train(self):
        # Training the Agent
        init = tf.initialize_all_variables()

        # Launch the tensorflow graph
        with tf.Session() as sess:
            sess.run(init)
            i = 0
            while i < self.total_episodes:

                #Choose either a random action or one from our network.
                if np.random.rand(1) < self.e:
                    action = np.random.randint(num_bandits)
                else:
                    action = sess.run(chosen_action)

                reward = self.pullBandit(action) #Get our reward from picking one of the bandits.

                #Update the network.
                # _, resp, ww = sess.run([update, responsible_weight, weights],
                #                     feed_dict={reward_holder:[reward], action_holder:[action]})
                resp, ww = self.Agent.update_network(reward, action)

                #Update our running tally of scores.
                self.total_reward[action] += reward

                if i % 50 == 0:
                    print("Running reward for the " + str(self.num_bandits) + " bandits: " + str(self.total_reward))
                i += 1

        print("The agent thinks bandit " + str(np.argmax(ww)+1) + " is the most promising....")

        if np.argmax(ww) == np.argmax(-np.array(self.Bandits.bandits)):
            print("...and it was right!")
        else:
            print("...and it was wrong!")

def main():
    tab = Two_Armed_Bandit()
    tab.train()

if __name__ == "__main__":
    main()

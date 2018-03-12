import gym
import numpy as np
import sys

class LeinforceRearning():
    def __init__(self, game_name="FrozenLake"):
        if game_name == "FrozenLake":
            self.env = gym.make("FrozenLake-v0")
        else:
            print("It is invalid game name.")
            return

        print("env.observation_space:{}".format(self.env.observation_space.n))
        print("env.action_space:{}".format(self.env.action_space.n))

        #create lists to contain total rewards and steps per episode
        self.jList = []
        self.rList = []

        self.s = None
        self.a = None
        self.s1 = None
        self.r = None

        self.d = None

    def set_learning_parameters(self):
        self.lr = 0.8
        self.y = 0.95
        self.num_episodes = 2000

    def initialize_Qtable_with_zeros(self):
        Q = np.zeros([self.env.observation_space.n, self.env.action_space.n])

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

    def chose_action_by_greedily_picking_from_Qtable(self, episode):
        #Choose an action by greedily (with noise) picking from Q table
        a = np.argmax(self.Q[self.s,:] + np.random.randn(1,self.env.action_space.n)*(1./(episode+1)))
        self.a = a
        return a

    def get_new_state_reward_from_environment(self, action):
        #Get new state and reward from environment
        s1, r, d, _ = self.env.step(action)
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

    def result(self, rList, Qtable=False):
        print("Score over time: " + str(sum(rList)/self.num_episodes))
        print("Step num over time: " + str(sum(self.jList)/self.num_episodes))
        if Qtable:
            print("Final Q-Table Values")
            print(self.Q)

    def train(self):
        Q = self.initialize_Qtable_with_zeros()
        self.set_learning_parameters()

        for i in range(self.num_episodes):
            #Reset environment and get first new observation
            s = self.env.reset() # start
            self.s = s

            # self.status_check(episode=i+1, step=0, Qtable=True)

            rAll = 0
            d = False
            j = 0

            #The Q-Table learning algorithm
            while j < 99: # step
                j+=1

                a = self.chose_action_by_greedily_picking_from_Qtable(episode=i)
                s1, r, d = self.get_new_state_reward_from_environment(action = a)
                Q = self.update_Qtable_with_new_knowledge(s, a, r, s1)

                # self.status_check(episode=i+1, step=j, Qtable=False)

                rAll += r
                self.s = s1
                s = s1
                if d == True:
                    break

            self.jList.append(j)
            self.rList.append(rAll)
            # self.result(self.rList, Qtable=False)

        self.result(self.rList, Qtable=False)

def main():
    LR = LeinforceRearning(game_name="FrozenLake")
    LR.train()


if __name__ == "__main__":
    main()

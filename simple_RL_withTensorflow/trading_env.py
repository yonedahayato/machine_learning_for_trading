import datetime
import itertools
import numpy as np
import pandas as pd
import pandas_datareader.data as web

class Observation:
    def __init__(self, stock_num, data_length):
        self.stock_num = stock_num
        self.data_length = data_length

        self.status = ["hold", "not_hold"]
        self.observation_num = stock_num * len(self.status) * data_length
        self.n = self.observation_num

    def make_status(self):
        self.hold_status = [0]*self.stock_num
        self.status_num = 0
        self.status_table = np.arange(self.observation_num).reshape(self.data_length, self.stock_num * len(self.status))

class Action:
    def __init__(self, stock_num):
        self.action_dic = {0:"buy", 1:"sell", 2:"hold"}
        self.n = len(self.action_dic) ** stock_num

        action_list = list(range(len(self.action_dic)))
        action_list = list(itertools.product(action_list,repeat=stock_num))

        self.action_table = np.array(action_list)
        print(self.action_table)

class Trading_Env:
    def __init__(self):
        stock_data_list = self.stock_data_load(view_data=False)

        self.observation_space = Observation(stock_num=len(stock_data_list), data_length=self.data_length)
        self.action_space = Action(stock_num=len(stock_data_list))

        self.observation_space.make_status()

    def stock_data_load(self, view_data=False):
        # start = datetime.datetime(2016,1,1)
        # end = datetime.date.today()
        start = "2016-01-01"
        end = "2017-01-01"

        nikkei225 = web.DataReader("NIKKEI225", "fred", start, end)
        Djia = web.DataReader("DJIA", "fred", start, end)
        status_df = pd.DataFrame(["not_hold"]*len(nikkei225), index=nikkei225.index, columns=["status"])

        nikkei225 = pd.concat([nikkei225, status_df], axis=1)
        nikkei225.columns = ["Close", "status"]
        Djia = pd.concat([Djia, status_df], axis=1)
        Djia.columns = ["Close", "status"]

        self.data_length = len(nikkei225)

        if view_data:
            print(nikkei225)

        stock_data_list = [Djia, nikkei225]
        self.stock_data_list = stock_data_list
        return stock_data_list

    def step(self, action):
        self.status_num += 1
        s1 = None
        r = None

        d = None
        _ = None

        return s1, r, d, _

    def reset(self):
        self.hold_status = [0]*len(self.stock_data_list)
        self.step_num = 0
        first_state = None
        return first_state

def main():
    TE = Trading_Env()

if __name__ == "__main__":
    main()

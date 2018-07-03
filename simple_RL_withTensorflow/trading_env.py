import datetime
import itertools
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import random
from sklearn import preprocessing
import talib

from helper.load_stock_data import Load_Stock_Data

class Observation:
    def __init__(self, train, status_value=False):
        self.train = train

        self.data_length = 0
        self.preparat_data_num = 10
        self.stock_data_list = []
        stock_data_list = self.load_stock_data(status_value)

        self.stock_num = len(stock_data_list)

        self.status = {0:"not_hold", 1:"hold"}

        self.observation_num = (len(self.status) ** self.stock_num) * self.data_length
        self.n = self.observation_num + 1

        self.penalty = None
        self.status_value = status_value

    def load_stock_data(self, status_value=False):
        # start = datetime.datetime(2016,1,1)
        # end = datetime.date.today()
        start = "2015-01-01"
        end = "2017-01-01"

        train_data_num = 200
        test_data_num = 200

        LSD = Load_Stock_Data()
        if status_value:
            train_data_num_all = train_data_num + self.preparat_data_num
            test_data_num_all = test_data_num + self.preparat_data_num
        else:
            train_data_num_all = train_data_num
            test_data_num_all = test_data_num

        stock_data_list_train, stock_data_list_test = \
            LSD.load_from_PandasDataReader(start, end, train_data_num_all, test_data_num_all, view_data=False)

        if self.train:
            # self.data_length = LSD.train_data_length
            self.data_length = train_data_num
            self.stock_data_list = stock_data_list_train
            return stock_data_list_train

        else:
            # self.data_length = LSD.test_data_length
            self.data_length = test_data_num
            self.stock_data_list = stock_data_list_test
            return stock_data_list_test

    def make_status(self):
        self.hold_status = [0]*self.stock_num
        self.status_table = np.arange(self.observation_num).reshape(self.data_length, len(self.status) ** self.stock_num)
        self.status_number_table = [list(value) for value in list(itertools.product([1, 0],repeat=self.stock_num))]

        if self.status_value:
            self.calculate_status()

    def update_status(self, action, Id, step_num):
        stock_data_df = self.stock_data_list[Id]

        if action == "buy":
            self.hold_status[Id] = 1
            # stock_data_df["status"].iloc[step_num] = "hold"
            stock_data_df["status"].iloc[step_num] = "buy"

        elif action == "sell":
            self.hold_status[Id] = 0
            # stock_data_df["status"].iloc[step_num] = "not_hold"
            stock_data_df["status"].iloc[step_num] = "sell"

        elif action in ["do_nothing", "too_much_buy", "too_much_sell"]:
            if not step_num == 0:
                before_status = stock_data_df["status"].iloc[step_num-1]
                if before_status == "buy":
                    stock_data_df["status"].iloc[step_num] = "hold"
                elif before_status == "sell":
                    stock_data_df["status"].iloc[step_num] = "not_hold"
                else:
                    stock_data_df["status"].iloc[step_num] = before_status
            else:
                stock_data_df["status"].iloc[step_num] = "not_hold"

            if action in ["too_much_buy", "too_much_sell"]:
                self.penalty = 100
                return

        self.penalty = 0
        return

    def update_all_status(self, actions_list, step_num):
        for Id in range(self.stock_num):
            action = actions_list[Id]
            hold_status = self.status[self.hold_status[Id]]

            if action == "buy" and hold_status == "not_hold":
                self.update_status(action="buy", Id=Id, step_num=step_num)

            elif action == "sell" and hold_status == "hold":
                self.update_status(action="sell", Id=Id, step_num=step_num)

            elif action == "do_nothing":
                self.update_status(action="do_nothing", Id=Id, step_num=step_num)

            else:
                if action == "buy" and hold_status == "hold":
                    self.update_status(action="too_much_buy", Id=Id, step_num=step_num)
                elif action == "sell" and hold_status == "not_hold":
                    self.update_status(action="too_much_sell", Id=Id, step_num=step_num)

        status_number = self.status_number_table.index(self.hold_status)

        if self.status_value:
            self.s1 = self.get_status_values(step_num)
        else:
            self.s1 = self.status_table[step_num, status_number]

        return

    def get_status(self):
        s1 = self.s1
        return s1

    def get_reward_info(self, step_num):
        r = 0
        for Id in range(len(self.stock_data_list)):
            stock_data_df = self.stock_data_list[Id]
            if stock_data_df["status"].iloc[step_num] == "sell":
                bought_time = stock_data_df.loc[stock_data_df["status"]=="buy", ["status"]].index

                buy_value = stock_data_df["Close"].loc[bought_time].values[-1]
                sell_value = stock_data_df["Close"].iloc[step_num]

                r_tmp = sell_value - buy_value
            else:
                r_tmp = 0
            r += r_tmp
        return r

    def reset(self):
        self.hold_status = [0]*self.stock_num
        for Id in range(self.stock_num):
            self.stock_data_list[Id].loc[:, "status"] = "not_hold"

    def calculate_status(self):
        msg = "[calculate_status]: "
        self.status_df_list = []
        # SMA : Simple Moving Average
        sma_list = []

        # Rate of deviation
        dr_list = []

        # Bollinger Band
        BBand_upper_list = []
        BBand_middle_list = []
        BBand_lower_list = []

        # PER
        ## todo ##

        for stock_cnt, stock_data_df in enumerate(self.stock_data_list):
            close_df = stock_data_df["Close"]

            sma_df = talib.SMA(close_df, timeperiod=self.preparat_data_num+1)
            sma_list.append(sma_df)

            dr_df = sma_df / close_df
            dr_list.append(dr_df)

            upper_df, middle_df, lower_df = talib.BBANDS(close_df, timeperiod=self.preparat_data_num+1)
            BBand_upper_list.append(upper_df)
            BBand_middle_list.append(middle_df)
            BBand_lower_list.append(lower_df)

            status_df = pd.concat([sma_df, dr_df, upper_df, middle_df, lower_df], axis=1)
            status_df.columns = ["sma", "rate_of_deviation", "BBand_upper", "BBand_middle", "BBand_lower"]
            status_df = (status_df - status_df.min()) / (status_df.max() - status_df.min())
            self.status_df_list.append(status_df)

            stock_data_df = pd.concat([stock_data_df, status_df], axis=1)
            stock_data_df = stock_data_df.iloc[self.preparat_data_num:]
            self.stock_data_list[stock_cnt] = stock_data_df
            self.max_status_num = 5

            # print(stock_data_df)

        return

    def get_status_values(self, step_num):
        s = []
        for stock_id in range(self.stock_num):
            status_df = self.status_df_list[stock_id]
            s = s + list(status_df.iloc[step_num].values)
        return s

class Action:
    def __init__(self, stock_num):
        self.action_dic = {0:"buy", 1:"sell", 2:"do_nothing"}
        self.stock_num = stock_num
        self.n = len(self.action_dic) ** stock_num

        action_list = list(range(len(self.action_dic)))
        action_list = list(itertools.product(action_list,repeat=stock_num))

        self.action_table = np.array(action_list)

    def parse_action(self, action):
        actions = self.action_table[action]

        actions_list = []
        for Id in range(self.stock_num):
            action_num = actions[Id]
            actions_list.append(self.action_dic[action_num])

        return actions_list

    def sample(self):
        action = random.randrange(self.n)
        return action

class Trading_Env:
    def __init__(self, train, status_value=False):
        self.train = train
        self.observation_space = Observation(train=train, status_value=status_value)
        self.observation_space.make_status()

        self.stock_num = self.observation_space.stock_num
        self.data_length = self.observation_space.data_length

        self.action_space = Action(stock_num=self.stock_num)

        self.step_num = -1
        self.s = -1

        self.status_value = status_value

    def calculate_reward(self, reward_info):
        r = reward_info
        return r

    def step(self, action):
        actions_list = self.action_space.parse_action(action)
        self.step_num += 1

        self.observation_space.update_all_status(actions_list, self.step_num)
        s1 = self.observation_space.get_status()

        reward_info = self.observation_space.get_reward_info(step_num = self.step_num)
        r = self.calculate_reward(reward_info)

        if self.step_num == self.data_length-1:
            d = True
            _ = None

            return s1, r, d, _

        d = None
        _ = None

        return s1, r, d, _

    def reset(self):
        self.hold_status = [0]*self.stock_num
        self.observation_space.reset()

        if self.status_value:
            self.step_num = 0
            s = self.observation_space.get_status_values(self.step_num)

            return s
        else:
            self.step_num = -1
            return -1

def main():
    TE = Trading_Env()

if __name__ == "__main__":
    main()

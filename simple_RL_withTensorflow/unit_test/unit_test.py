import os
import pandas as pd
import pytest
import shutil
import sys
from time import sleep
from unittest import TestCase, main
import unittest

pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)
sys.path.extend(["../helper", "./helper"])
from trading_env import Trading_Env, Observation, Action
from Q_Learning_with_Tables_and_NN import ReinforceLearning, ReinforceLearning_NN
from load_stock_data import Load_Stock_Data

class Unit_Test(TestCase):
    @classmethod
    def setUpClass(cls):
        print('*** 全体前処理 ***')
        cls.one = 1
        cls.check_data_dict = {"data_name": [], "data":[]}
        # test_name.check_data_dict["data_name"].append("data_name1")
        # test_name.check_data_dict["data"].append(data1)
        cls.observation_space_train = Observation(train=True, status_value=True)
        cls.observation_space_test = Observation(train=False, status_value=True)

    def setUp(self):
        print('+ テスト前処理')

    def tearDown(self):
        print('+ テスト後処理')

    @classmethod
    def tearDownClass(cls):
        print('*** 全体後処理 ***')

        if len(cls.check_data_dict["data_name"]) == len(cls.check_data_dict["data"]) and \
            len(cls.check_data_dict["data_name"]) != 0:

            for Index in range(len(cls.check_data_dict["data_name"])):
                print(cls.check_data_dict["data_name"][Index])
                print(cls.check_data_dict["data"][Index])

            # cls.compare_data("data_name1", "data_name2")

        else:
            print("check_data`s is invalid length.")

    @classmethod
    def compare_data(self, data_name1, data_name2):
        data1, data2 = None, None
        for Index in range(len(cls.check_data_dict["data_name"])):
            if cls.check_data_dict["data_name"][Index] == "data_name1":
                data1 = cls.check_data_dict["data"][Index]
            if cls.check_data_dict["data_name"][Index] == "data_name2":
                data2 = cls.check_data_dict["data"][Index]

        if data1 == None or data2 == None:
            print("fail to compare")

        else:
            print("compare_check:{}".format(data1 == data2))

    @unittest.skip("tmp")
    def test_load_stock_data(self):
        env = Trading_Env(train=True)
        stock_data_list = env.observation_space.stock_data_list
        size_list = []
        for stock_data in stock_data_list:
            size_list.append(stock_data.shape)

        print(size_list)

    @unittest.skip("tmp")
    def test_load_stock_data_from_CSVfile_case1(self):
        save_path = "./helper/stock_data"
        if os.path.exists(save_path):
            shutil.rmtree(save_path)

        LSD = Load_Stock_Data(save=True)

        start = "2015-01-01"
        end = "2017-01-01"

        train_data_num = 200
        test_data_num = 200

        stock_data_list_train, stock_data_list_test = \
            LSD.load_from_PandasDataReader(start, end, train_data_num, test_data_num, view_data=False)

        LSD_formCSV = Load_Stock_Data(save=True)

        stock_data_list_train, stock_data_list_test = LSD_formCSV.load_from_CSVfiles()

    def check_profit_result(self, stock_data_df):
        status = "not_hold"
        error_list = []
        error_template = "Id: {}, status: {}, new_status: {}"
        success_cnt = 0

        buy_value = 0
        close = 0
        profit = 0

        for Id in range(len(stock_data_df)):
            new_status = stock_data_df["status"].iloc[Id]
            new_close = stock_data_df["Close"].iloc[Id]

            if status == "not_hold" and new_status in ["hold", "sell"]:
                error_list.append(error_template.format(Id, status, new_status))
            elif status == "hold" and new_status in ["not_hold", "buy"]:
                error_list.append(error_template.format(Id, status, new_status))
            elif status == "buy" and new_status in ["not_hold", "buy"]:
                error_list.append(error_template.format(Id, status, new_status))
            elif status == "sell" and new_status in ["hold", "sell"]:
                error_list.append(error_template.format(Id, status, new_status))
            else:
                success_cnt += 1
                if new_status == "buy":
                    buy_value = new_close
                elif new_status == "sell":
                    profit += (new_close - buy_value)

            status = new_status
            close = new_close

        if success_cnt != len(stock_data_df):
            error_list.append("error: success_cnt != len(stock_data_df)")

        return profit, error_list

    def test_check_profit_result(self):
        RL = ReinforceLearning(game_name="Trading")
        RL.set_parameters(num_episodes=3)
        RL.train()
        RL.test()
        RL.result(Qtable=False, check=False, train=True)
        pd.options.display.max_rows = 1000

        error_list = []
        error_cnt = 0
        profit = 0
        last_stock_data_list = RL.train_env.observation_space.stock_data_list

        for stock_id, stock_data_df in enumerate(last_stock_data_list):
            profit_tmp, error_list_tmp = self.check_profit_result(stock_data_df)

            profit += profit_tmp
            error_list.append(error_list_tmp)
            if len(error_list_tmp) != 0:
                error_cnt += 1

        if (error_cnt != 0) or (RL.rList[-1] != profit):
            err_msg = "error, error_cnt: {}, RL.rList[-1]: {}, profit: {}".format(error_cnt, RL.rList[-1], profit)

            Unit_Test.check_data_dict["data_name"].append(err_msg)
            Unit_Test.check_data_dict["data"].append(error_list)

            Unit_Test.check_data_dict["data_name"].append("last_stock_data_list")
            Unit_Test.check_data_dict["data"].append(last_stock_data_list)

            result = False
        else:
            result = True

        self.assertTrue(result)

    @unittest.skip("tmp")
    def test_frozenlake(self):
        RL = ReinforceLearning(game_name="FrozenLake")
        RL.train()
        RL.test()
        # RL.result(Qtable=False, check=False, train=True)

    def test_frozenlake_NN(self):
        RL_NN = ReinforceLearning_NN(game_name="FrozenLake")
        RL_NN.set_parameters(num_episodes=3)
        for i in range(3):
            print("== {} ==".format(i))
            RL_NN.train()
            RL_NN.test()

    def test_Trading_NN_not_status_value(self):
        for i in range(3):
            print("== {} ==".format(i))
            RL_NN = ReinforceLearning_NN(game_name="Trading", status_value=False)
            RL_NN.set_parameters(num_episodes=3)
            RL_NN.train()
            RL_NN.test()

    def test_calculate_status(self):
        Unit_Test.observation_space_train.calculate_status()

    @pytest.mark.value
    def test_Trading_NN_status_value(self):
        RL_NN_value = ReinforceLearning_NN(game_name="Trading", status_value=True)
        RL_NN_value.set_parameters(num_episodes=3)
        RL_NN_value.train()
        RL_NN_value.test()

    @unittest.skip("skip message <skipもできる>")
    def test_skip(self):
        print("skip")

    @unittest.skip("skip message <skipもできる>")
    def test_subTest(self):
        def same_value(x):
            return x

        for i in range(10):
            with self.subTest(arg_1=1, arg2=2):
                self.assertEqual(same_value(i), i)

if __name__ == "__main__":
    unittest.main()

# pytest unit_test.py -v -n 2

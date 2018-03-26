import os
import sys
from unittest import TestCase, main
import unittest

pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)
from trading_env import Trading_Env
from Q_Learning_with_Tables_and_NN import LeinforceRearning

class test_name(TestCase):
    @classmethod
    def setUpClass(cls):
        print('*** 全体前処理 ***')
        cls.one = 1
        cls.check_data_dict = {"data_name": [], "data":[]}
        # test_name.check_data_dict["data_name"].append("data_name1")
        # test_name.check_data_dict["data"].append(data1)

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

    def test_load_stock_data(self):
        env = Trading_Env(train=True)
        stock_data_list = env.observation_space.stock_data_list
        size_list = []
        for stock_data in stock_data_list:
            size_list.append(stock_data.shape)
            print(stock_data)

        print(size_list)

    def test_check_profit_result(self):
        LR = LeinforceRearning(game_name="Trading")
        LR.set_parameters(num_episodes=3)
        LR.train()
        print(LR.rList)

    @unittest.skip("skip message <skipもできる>")
    def test_skip(self):
        print("skip")

    def test_subTest(self):
        def same_value(x):
            return x

        for i in range(10):
            with self.subTest(arg_1=1, arg2=2):
                self.assertEqual(same_value(i), i)

if __name__ == "__main__":
    unittest.main()

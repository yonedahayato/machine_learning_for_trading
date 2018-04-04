import os
import pandas as pd
import pandas_datareader.data as web

class Load_Stock_Data():
    def __init__(self, save=False):
        self.save = save
        self.save_path, self.save_train_data_path, self.save_test_data_path = "", "", ""
        self.setting_save_path(save=save)

        self.train_data_length, self.test_data_length = 0, 0
        self.stock_data_list_train, self.stock_data_list_test = [], []

    def setting_save_path(self, save):
        if os.path.exists("./helper"):
            self.save_path = "./helper/stock_data"
        else:
            self.save_path = "./stock_data"

        self.save_train_data_path = self.save_path + "/train"
        self.save_test_data_path = self.save_path + "/test"

        if save:
            path_list = [self.save_path, self.save_train_data_path, self.save_test_data_path]
            for path in path_list:
                if not os.path.exists(path):
                    os.mkdir(path)

    def make_file_path(self, file_name, train):
        if train:
            path = self.save_train_data_path + "/" + file_name + ".csv"
        else:
            path = self.save_test_data_path + "/" + file_name + ".csv"
        return path

    def save_ToCSV(self, data_df, data_name, train):
        save_path = self.make_file_path(data_name, train=train)

        data_df.to_csv(save_path)

    def save_ToCSVs(self, DF_list, data_name_list, train):
        if len(DF_list) != len(data_name_list):
            raise("[load_stock_data]: DF_list != data_name_list")

        for data_df, data_name in zip(DF_list, data_name_list):
            self.save_ToCSV(data_df, data_name, train)

    def load_from_PandasDataReader(self, start, end, train_data_num, test_data_num, view_data=False):
        nikkei225 = web.DataReader("NIKKEI225", "fred", start, end)
        nikkei225 = nikkei225.fillna(method='ffill')
        nikkei225 = nikkei225.fillna(nikkei225.mean())

        Djia = web.DataReader("DJIA", "fred", start, end)
        Djia = Djia.fillna(method='ffill')
        Djia = Djia.fillna(Djia.mean())

        status_df = pd.DataFrame(["not_hold"]*len(nikkei225), index=nikkei225.index, columns=["status"])

        nikkei225 = pd.concat([nikkei225, status_df], axis=1)
        nikkei225.columns = ["Close", "status"]
        Djia = pd.concat([Djia, status_df], axis=1)
        Djia.columns = ["Close", "status"]

        nikkei225_train = nikkei225.iloc[:train_data_num]
        nikkei225_test = nikkei225.iloc[train_data_num:train_data_num+test_data_num]
        Djia_train = Djia.iloc[:train_data_num]
        Djia_test = Djia.iloc[train_data_num:train_data_num+test_data_num]

        self.train_data_length = len(nikkei225_train)
        self.test_data_length = len(nikkei225_test)

        if view_data:
            print(nikkei225)

        stock_data_list_train = [Djia_train, nikkei225_train]
        stock_data_list_test = [Djia_test, nikkei225_test]

        self.stock_data_list_train = stock_data_list_train
        self.stock_data_list_test = stock_data_list_test

        if self.save:
            self.save_ToCSVs(stock_data_list_train, ["Djia_train", "nikkei225_train"], train=True)
            self.save_ToCSVs(stock_data_list_test, ["Djia_test", "nikkei225_test"], train=False)

        return stock_data_list_train, stock_data_list_test

    def load_from_CSVfile(self, data_name, train):
        load_path = self.make_file_path(data_name, train=train)

        data_df = pd.read_csv(load_path, header=1, index_col=1)
        return data_df

    def load_from_CSVfiles(self, train_data_name_list=["Djia_train", "nikkei225_train"],
                            test_data_name_list=["Djia_test", "nikkei225_test"]):

        for train in [True, False]:
            if train:
                data_name_list = train_data_name_list
            else:
                data_name_list = test_data_name_list

            for data_name in data_name_list:
                data_df = self.load_from_CSVfile(data_name, train=train)

                if train:
                    self.train_data_length = len(data_df)
                    self.stock_data_list_train.append(data_df)
                else:
                    self.test_data_length = len(data_df)
                    self.stock_data_list_test.append(data_df)

        return self.stock_data_list_train, self.stock_data_list_test

def load_stock_data():
    LSD = Load_Stock_Data()

    start = "2015-01-01"
    end = "2017-01-01"

    train_data_num = 200
    test_data_num = 200

    stock_data_list_train, stock_data_list_test = \
        LSD.load_from_PandasDataReader(start, end, train_data_num, test_data_num, view_data=True)

if __name__ == "__main__":
    load_stock_data()

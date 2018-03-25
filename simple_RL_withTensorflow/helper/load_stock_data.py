import pandas as pd
import pandas_datareader.data as web

class Load_Stock_Data():
    def __init__(self):
        pass

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
        return stock_data_list_train, stock_data_list_test

    def load_from_CSVfile():
        pass

def load_stock_data():
    LSD = Load_Stock_Data()

    start = "2015-01-01"
    end = "2017-01-01"

    train_data_num = 200
    test_data_num = 200

    stock_data_list_train, stock_data_list_test = \
        LSD.load_from_PandasDataReader(start, end, train_data_num, end_data_num, view_data=True)

if __name__ == "__main__":
    load_stock_data()

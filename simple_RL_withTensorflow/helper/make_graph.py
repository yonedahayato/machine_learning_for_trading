from datetime import datetime as dt
import matplotlib.pyplot as plt
import os
import pandas as pd

class Make_Graph:
    def __init__(self, file_name="", Id_name="Id", value_name="value"):
        self.file_name = file_name
        self.Id_name = Id_name
        self.value_name = value_name

        self.setting_save_path()

        self.Id_list = []
        self.value_list = []

    def setting_save_path(self):
        if os.path.exists("./helper"):
            self.save_path = "./helper/graph"
        else:
            self.save_path = "./graph"

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

    def data_input(self, Id, value):
        self.Id_list.append(Id)
        self.value_list.append(value)

    def save_graph(self, fig):
        now = dt.now()
        now_str = now.strftime("%Y-%m-%d-%H-%M")
        file_name = self.save_path + "/" + self.file_name + "_" + now_str + ".png"

        fig.savefig(file_name)

    def save_line_graph(self):
        data_df = pd.DataFrame({self.Id_name: self.Id_list, self.value_name: self.value_list})
        data_df = data_df.set_index(self.Id_name)

        ax = data_df.plot()
        fig = ax.get_figure()

        self.save_graph(fig)

    def save_chart_graph(self, stock_data_df):
        close_data_df = stock_data_df[["Close"]]

        ax = close_data_df.plot()
        fig = ax.get_figure()

        buy_point = stock_data_df.loc[stock_data_df["status"]=="buy"].index
        buy_value = stock_data_df.loc[stock_data_df["status"]=="buy", "Close"]

        sell_point = stock_data_df.loc[stock_data_df["status"]=="sell"].index
        sell_value = stock_data_df.loc[stock_data_df["status"]=="sell", "Close"]

        ax.scatter(list(buy_point), list(buy_value), label="buy", c="r", marker="^")
        ax.scatter(list(sell_point), list(sell_value), label="sell", c="b", marker="v")
        ax.legend()

        self.save_graph(fig)

def make_graph():
    mg = Make_Graph(file_name="test_file", Id_name="ID", value_name="Val")
    for Id in range(10):
        value = Id * 3
        mg.data_input(Id=Id, value=value)
    mg.save_line_graph()

if __name__ == "__main__":
    make_graph()

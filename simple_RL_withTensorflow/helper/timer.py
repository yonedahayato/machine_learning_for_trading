import copy
import csv
from datetime import datetime as dt
import os
import time

class Timer():
    def __init__(self):
        self.start_time = ""

        self.result_dict_tmp = {"name": "", "start_time": "", "elapsed_time": ""} # deepcopyで使用
        self.result_name_list = []
        self.results_list = []

        self.result_path = ""
        self.setting_result_path()

    def setting_result_path(self):
        if os.path.exists("./helper"):
            self.result_path = "./helper/time_measure_result"
        else:
            self.result_path = "./time_measure_result"

        if not os.path.exists(self.result_path):
            os.mkdir(self.result_path)

    def start(self, name=None):
        if "name" == None:
            raise("[Timer, start]: name is invalid.")

        result_dict = copy.deepcopy(self.result_dict_tmp)

        name = str(name)
        result_dict["name"] = name
        self.result_name_list.append(name)

        result_dict["start_time"] = time.time()
        self.results_list.append(result_dict)

    def stop(self, name=None):
        name = str(name)

        if (name == None) or (name not in self.result_name_list):
            print("name list")
            print(self.result_name_list)
            raise("[Timer, stop]: name is invalid.")

        Index = self.result_name_list.index(name)
        result_dict = self.results_list[Index]
        result_dict["elapsed_time"] = time.time() - result_dict["start_time"]

    def result_print(self):
        if self.results_list == []:
            print("there are no results")
            return

        for Id in range(len(self.results_list)):
            result_dict = self.results_list[Id]
            name = result_dict["name"]
            elapsed_time = result_dict["elapsed_time"]
            print("name: {}, elapsed_time: {}".format(name, elapsed_time))

    def result_write_csv(self):
        now = dt.now()
        now_str = now.strftime("%Y-%m-%d-%H-%M")
        file_name = self.result_path + "/" + now_str + ".csv"
        with open(file_name, "w") as f:
            writer = csv.DictWriter(f, self.result_dict_tmp.keys())
            writer.writeheader()
            for result_dict in self.results_list:
                writer.writerow(result_dict)

def time_measure():
    timer = Timer()

    for cnt in range(5):
        timer.start(name=cnt)
        time.sleep(1)
        timer.stop(name=cnt)

    timer.result_print()
    timer.result_write_csv()

if __name__ == "__main__":
    time_measure()

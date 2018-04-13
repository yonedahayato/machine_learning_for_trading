from datetime import datetime as dt
from glob import glob
import os
import sys
import tensorflow as tf

sys.path.append("./helper")
from params_setting import load_file

class Setting:
    def __init__(self, save_flag=True):
        self.save_flag = save_flag
        self.setting_params_path()

        self.saver = tf.train.Saver()

    def setting_params_path(self):
        if os.path.exists("./helper"):
            self.params_path = "./helper/params"
        else:
            self.params_path = "./params"

        if not os.path.exists(self.params_path):
            if not self.save_flag: # load
                raise Exception("[setting params path]: There are not load directory.")
            else: # save
                os.mkdir(self.params_path)

class Save_Params(Setting):
    def __init__(self):
        Setting.__init__(self, save_flag=True)

        self.setting_save_path()

    def setting_save_path(self):
        now = dt.now()
        now_str = now.strftime("%Y-%m-%d-%H-%M-%S")

        self.save_path = self.params_path + "/" + now_str

    def save(self, sess, file_name):
        self.save_path = self.save_path + "/" + file_name
        print("[save]: save parameters, {}".format(self.save_path))
        self.saver.save(sess, self.save_path)

        return file_name

class Load_Params(Setting):
    def __init__(self):
        Setting.__init__(self, save_flag=False)

    def find_latest_params(self, file_name):
        file_name = file_name.split("/")[-1]

        params_dir_list = glob(self.params_path + "/*/")
        params_dir_list = sorted(params_dir_list, reverse=True)

        if len(params_dir_list) == 0:
            raise Exception("[find latest params]: can not find params files in {}".format(self.params_path))

        for params_dir in params_dir_list:
            load_path = params_dir + file_name
            if self.check_exists_params_files(load_path):
                return load_path
        raise Exception("[find latest params]: can not find latest params file")

    def check_exists_params_files(self, load_path):
        check = os.path.exists(load_path + ".index")
        return check

    def load(self, sess, file_name=""):
        if file_name == "":
            file_name = load_file

        self.load_path = self.params_path + "/" + file_name

        if not self.check_exists_params_files(self.load_path):
            print("[load]: can not find this params file, {}".format(self.load_path))
            print("[load]: finding latest params...")
            self.load_path = self.find_latest_params(file_name)

        print("[load]: loading params, {}".format(self.load_path))
        self.saver.restore(sess, self.load_path)

def save_load():
    from tensorflow.examples.tutorials.mnist import input_data

    data_dir = "./data/mnist/"
    sess_file = "2018-04-10-12-56-58/test.ckpt"
    imagesize = 28
    n_label = 10
    n_batch = 100
    n_train = 1000
    learning_rate = 0.5

    mnist = input_data.read_data_sets(data_dir, one_hot=True)

    x = tf.placeholder(tf.float32, [None, imagesize ** 2])
    W = tf.get_variable("W", [imagesize ** 2, n_label], initializer=tf.random_normal_initializer())
    b = tf.get_variable("b", [n_label], initializer=tf.constant_initializer(0.0))
    y = tf.matmul(x, W) + b

    y_ = tf.placeholder(tf.float32, [None, n_label])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sp = Save_Params()
    lp = Load_Params()

    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess_path = os.path.join(sess_dir, sess_file)
        try:
            lp.load(sess, file_name=sess_file)
        except Exception as e:
            print(e)
            sess.run(tf.global_variables_initializer())

        for _ in range(n_train):
            batch_xs, batch_ys = mnist.train.next_batch(n_batch)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        result = sess.run(accuracy, feed_dict={x: mnist.test.images,
                                            y_: mnist.test.labels})
        print('Accuracy => ' + str(result))

        sp.save(sess, file_name="test.ckpt")

if __name__ == "__main__":
    save_load()

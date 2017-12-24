import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spo

def f(x):
    y = (x - 1.5)**2 + 0.5
    print("x = {}, y = {}".format(x,y))
    return y

def error(line, data):
    err = np.sum((data[:, 1] - (line[0] * data[:, 0] + line[1]))**2)
    return err

def fit_line(data, error_func):
    l = np.float32([0, np.mean(data[:, 1])])

    x_ends = np.float32([-5, 5])
    plt.plot(x_ends, l[0] * x_ends + l[1], "m--", linewidth=2.0, label="Initial guesess")

    result = spo.minimize(error_func, l, args=(data, ), method="SLSQP", options={"disp": True})
    return result.x

def error_poly(C, data):
    err = np.sum((data[:, 1] - np.polyval(C, data[:, 0])) ** 2)
    return err

def fit_poly(data, error_func, degree=3):
    Cguess = np.poly1d(np.ones(degree+1, dtype=np.float32))

    x= np.linspace(-5, 5, 21)
    plt.plot(x, np.polyval(Cguess, x), "m--", linewidth=2.0m label="Intitial guess")

    result = spo.minimize(error_func, Cguess, args=(data, ), method="SLSQP", options={"disp": True})
    return np.poly1d(result.x)

def test_run():
    l_orig = np.float32([4, 2])
    print("original line: C0 = {}, C1 = {}".format(l_orig[0], l_orig[1]))
    Xorig = np.linspace(0, 10, 21)
    Yorig = l_orig[0] * Xorig + l_orig[1]
    plt.plot(Xorig, Yorig, "b--", linewidth=2.0, label="Orinal line")

    noise_sigma = 3.0
    noise = np.random.normal(0, noise_sigma, Yorig.shape)
    data = np.asarray([Xorig, Yorig + noise]).T
    plt.plot(data[:, 0], data[:, 1], "go", label="Data points")

    l_fit = fit_line(data, error)
    print("Fitted line: C0 = {}, C1 = {}".format(l_fit[0], l_fit[1]))
    plt.plot(data[:, 0], l_fit[0] * data[:, 0] + l_fit[1], "r--", linewidth=2.0)

    plt.show()
if __name__ == "__main__":
    test_run()

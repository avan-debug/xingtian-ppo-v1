import numpy as np
import pandas as pd


def ka_fang(table):
    n = np.sum(table)
    xi = np.sum(table, axis=1)
    yi = np.sum(table, axis=0)
    freq_table = np.matmul(xi.reshape(-1, 1), yi.reshape(1, -1))
    ka_fang = n * np.sum((table - freq_table / n) ** 2 / freq_table)
    return ka_fang


def main():
    X = np.array([68, 53, 70, 84, 60, 72, 51, 83, 70, 64])
    Y = np.array([288, 293, 349, 343, 290, 354, 283, 324, 340, 286])
    x_ = np.mean(X)
    y_ = np.mean(Y)

    pass


if __name__ == '__main__':
    # main()
    # table = np.array([
    #     [73, 32, 74, 21],
    #     [59, 16, 65, 20],
    #     [48, 12, 51, 29],
    # ])
    # table = np.array([
    #     [74, 67, 27],
    #     [11, 14, 33],
    #     [5, 10, 17],
    # ])
    # print("x^2 = {:.2f}".format(ka_fang(table)))
    a = np.random.randint(0, 4, 1)
    b = np.eye(4)[a]
    c = np.expand_dims(b, 0)
    d = np.append(c, [b], axis=0)
    print(a, b, c, d, sep="\n")
    print(a.shape, b.shape, c.shape, d.shape, sep="; ")

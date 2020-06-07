# %%

import pandas as pd
import matplotlib.pyplot as plt
import math

train_data_ = pd.read_csv('dataset/train.data', sep='\s+', header=None)
test_data_ = pd.read_csv('dataset/test.data', sep='\s+', header=None)

train_data_.rename(columns={10: 'target'}, inplace=True)
test_data_.rename(columns={10: 'target'}, inplace=True)


def split_xy(data):
    _x = data.drop('target', axis=1)
    _y = data['target']
    return _x, _y


_x, _y = split_xy(train_data_)

# %%
for i in range(10):
    for j in range(i + 1, 10):
        plt.clf()
        _i, _j = train_data_.iloc[:, i], train_data_.iloc[:, j]
        plt.scatter(_i[_y == 1], _j[_y == 1], zorder=2)
        plt.scatter(_i[_y == -1], _j[_y == -1], zorder=1)
        plt.savefig(f"2dout/fig{i}_{j}.png")

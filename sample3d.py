
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import auc, classification_report, f1_score, precision_score, recall_score, roc_auc_score, roc_curve

train_data_ = pd.read_csv('dataset/train.data', sep='\s+', header=None)
test_data_ = pd.read_csv('dataset/test.data', sep='\s+', header=None)

train_data_.rename(columns={10: 'target'}, inplace=True)
test_data_.rename(columns={10: 'target'}, inplace=True)

x = np.squeeze(train_data_.iloc[:, 1])
y = np.squeeze(train_data_.iloc[:, 0])
z = np.squeeze(train_data_.iloc[:, 2])
col = np.squeeze(train_data_['target'])


# 绘制散点图
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x, y, z, c=col)


# 展示
plt.show()

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import auc, classification_report, f1_score, precision_score, recall_score, roc_auc_score, roc_curve

train_data_ = pd.read_csv('dataset/train.data', sep='\s+', header=None)
test_data_ = pd.read_csv('dataset/test.data', sep='\s+', header=None)

train_data_.rename(columns={10: 'target'}, inplace=True)
test_data_.rename(columns={10: 'target'}, inplace=True)

for i in range(10):
    for j in range(i+1, 10):
        ax = train_data_.plot.scatter(x=i,
                                      y=j,
                                      c='target',
                                      colormap='viridis')
        plt.savefig(f"out/fig{i}_{j}.png")

# %%
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

train_data_ = pd.read_csv('dataset/train.data', sep='\s+', header=None)
test_data_ = pd.read_csv('dataset/test.data', sep='\s+', header=None)

train_data_.rename(columns={10:'target'},inplace=True)
test_data_.rename(columns={10:'target'},inplace=True)

# %%
target_count = train_data_.target.value_counts()
print('Class -1:', target_count[-1])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[-1] / target_count[1], 2), ': 1')
print(f'Rate: {target_count[-1]/(target_count[-1] + target_count[1])*100}%')

target_count.plot(kind='bar', title='Count (target)')

# %%
count_class_0, count_class_1 = train_data_.target.value_counts()

#%%

# Divide by class
df_class_0 = train_data_[train_data_.target == -1]
df_class_1 = train_data_[train_data_.target == 1]

#%%
df_class_0_under = df_class_0.sample(count_class_1)
df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)

# 欠采样
print('Random under-sampling:')
print(df_test_under.target.value_counts())

df_test_under.target.value_counts().plot(kind='bar', title='Count (target)')

#%%

# 超采样
df_class_1_over = df_class_1.sample(count_class_0, replace=True)
df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)

print('Random over-sampling:')
print(df_test_over.target.value_counts())

df_test_over.target.value_counts().plot(kind='bar', title='Count (target)');



# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

train_data_ = pd.read_csv('dataset/train.data', sep='\s+', header=None)
test_data_ = pd.read_csv('dataset/test.data', sep='\s+', header=None)

train_data_.rename(columns={10: 'target'}, inplace=True)
test_data_.rename(columns={10: 'target'}, inplace=True)


def split_xy(data):
    _x = data.drop('target', axis=1)
    _y = data['target']
    return _x, _y


# %%
# test 数据分布
target_count = test_data_.target.value_counts()
print('Class -1:', target_count[-1])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[-1] / target_count[1], 2), ': 1')
print(f'Rate: {target_count[-1] / (target_count[-1] + target_count[1]) * 100}%')

plt.cla()
_plt_x = 6
plt.figure(figsize=(_plt_x, _plt_x / 1.5))
ax = target_count.plot(kind='bar', title='Count (target Class)', rot=0)
ax.set_ylim([0, 2200])
for p in ax.patches:
    ax.annotate(np.round(p.get_height(), decimals=2), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center',
                va='center', xytext=(0, 10), textcoords='offset points')
plt.savefig('test_dist.png', bbox_inches='tight', dpi=400, pad_inches=0.05)

# %%
# train 数据分布
target_count = train_data_.target.value_counts()
print('Class -1:', target_count[-1])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[-1] / target_count[1], 2), ': 1')
print(f'Rate: {target_count[-1] / (target_count[-1] + target_count[1]) * 100}%')

plt.cla()
_plt_x = 6
plt.figure(figsize=(_plt_x, _plt_x / 1.5))
ax = target_count.plot(kind='bar', title='Count (target Class)', rot=0)
ax.set_ylim([0, 8500])
for p in ax.patches:
    ax.annotate(np.round(p.get_height(), decimals=2), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center',
                va='center', xytext=(0, 10), textcoords='offset points')
plt.savefig('train_dist.png', bbox_inches='tight', dpi=400, pad_inches=0.05)

# %%
# 分割数据，正类和负类
count_class_0, count_class_1 = train_data_.target.value_counts()

# Divide by class
df_class_0 = train_data_[train_data_.target == -1]
df_class_1 = train_data_[train_data_.target == 1]

# %%

# 欠采样（丢失数据不好）
df_class_0_under = df_class_0.sample(count_class_1)
df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)

print('Random under-sampling:')
print(df_test_under.target.value_counts())

df_test_under.target.value_counts().plot(kind='bar', title='Count (target)')

# %%

# 超采样
df_class_1_over = df_class_1.sample(count_class_0, replace=True)
df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)
df_test_over.drop(index=1, axis=0)

print('Random over-sampling:')
print(df_test_over.target.value_counts())

df_test_over.target.value_counts().plot(kind='bar', title='Count (target)')

# %%
# RF

over_x, over_y = split_xy(df_test_over)

X_train, X_test, y_train, y_test = train_test_split(over_x, over_y, test_size=0.2, random_state=42)

# %%
rfc = RandomForestClassifier(n_estimators=100, random_state=42)  # criterion = entopy,gini

n_est = [3, 10, 15, 40, 50, 60]
rfc_grid = [{'n_estimators': n_est,
             'max_depth': [1, 2, 3, 4, 5, 6, 7, None]},
            {'bootstrap': [False], 'n_estimators': n_est,
             'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}]
rfc_gs = GridSearchCV(estimator=rfc, param_grid=rfc_grid, cv=10, scoring='f1', n_jobs=-1)
rfc_gs.fit(over_x, over_y)
# rfc_f1_scores = cross_val_score(estimator=rfc_gs, X=over_x, y=over_y, scoring='f1', cv=5)
# print('CV clf f1 score:%.3f +/- %.3f' % (np.mean(rfc_f1_scores), np.std(rfc_f1_scores)))

print(rfc_gs.best_params_)  # {'bootstrap': False, 'max_depth': None, 'n_estimators': 10}
print(rfc_gs.best_score_)  # 0.9836533023676681


# %%

def print_report(data_name, clf_name, clf, x_test, y_real):
    print("-" * 20 + f"{data_name}@{clf_name}" + "-" * 20)
    y_pred = clf.predict(x_test)
    print(classification_report(y_real, y_pred))


# 精度 train
test_x, test_y = split_xy(train_data_)
print_report('TRAIN', 'RF', rfc_gs, test_x, test_y)

# 精度 test
test_x, test_y = split_xy(test_data_)
print_report('TEST', 'RF', rfc_gs, test_x, test_y)

# %%

# %%
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import auc, classification_report, f1_score, precision_score, recall_score, roc_auc_score, roc_curve

train_data_ = pd.read_csv('dataset/train.data', sep='\s+', header=None)
test_data_ = pd.read_csv('dataset/test.data', sep='\s+', header=None)

train_data_.rename(columns={10:'target'},inplace=True)
test_data_.rename(columns={10:'target'},inplace=True)

# %%
# test 数据分布
target_count = test_data_.target.value_counts()
print('Class -1:', target_count[-1])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[-1] / target_count[1], 2), ': 1')
print(f'Rate: {target_count[-1]/(target_count[-1] + target_count[1])*100}%')

ax = target_count.plot(kind='bar', title='Count (target Class)', rot=0)
ax.set_ylim([0, 2200])
for p in ax.patches:
    ax.annotate(np.round(p.get_height(), decimals=2), (p.get_x()+p.get_width()/2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
plt.savefig('test_dist.png',bbox_inches='tight',dpi=400,pad_inches=0.05)

# %%
# train 数据分布
target_count = train_data_.target.value_counts()
print('Class -1:', target_count[-1])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[-1] / target_count[1], 2), ': 1')
print(f'Rate: {target_count[-1]/(target_count[-1] + target_count[1])*100}%')

ax = target_count.plot(kind='bar', title='Count (target Class)', rot=0)
ax.set_ylim([0, 8500])
for p in ax.patches:
    ax.annotate(np.round(p.get_height(), decimals=2), (p.get_x()+p.get_width()/2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
plt.savefig('train_dist.png', bbox_inches='tight', dpi=400, pad_inches=0.05)

# %%
# 分割数据，正类和负类
count_class_0, count_class_1 = train_data_.target.value_counts()

# Divide by class
df_class_0 = train_data_[train_data_.target == -1]
df_class_1 = train_data_[train_data_.target == 1]

#%%

# 欠采样
df_class_0_under = df_class_0.sample(count_class_1)
df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)

print('Random under-sampling:')
print(df_test_under.target.value_counts())

df_test_under.target.value_counts().plot(kind='bar', title='Count (target)')

#%%

# 超采样
df_class_1_over = df_class_1.sample(count_class_0, replace=True)
df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)

print('Random over-sampling:')
print(df_test_over.target.value_counts())

df_test_over.target.value_counts().plot(kind='bar', title='Count (target)')

#%%
# RF
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score

over_x = df_test_over.drop('target', axis = 1)
over_y = df_test_over['target']

X_train, X_test, y_train, y_test = train_test_split(over_x,over_y, test_size = 0.2, random_state = 42)
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

#%%
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 100, random_state = 42)#criterion = entopy,gini
# rfc.fit(X_train, y_train)
# rfcpred = rfc.predict(X_test)
# RFCCV = (cross_val_score(rfc, X_train, y_train, cv=k_fold, scoring = 'accuracy').mean())

# models = pd.DataFrame({
#                 'Models': ['Random Forest Classifier'],
#                 'Score':  [RFCCV]})

# models.sort_values(by='Score', ascending=False)

rfc_grid = [{'n_estimators': [3, 10, 15, 40,50,60,70,80,90,100,120,150,170,200], 'max_depth':[1, 2, 3, 4, 5, 6, 7, None]},
            {'bootstrap': [False], 'n_estimators':[3, 10, 15, 40,50,60,70,80,90,100,120,150,170,200], 'max_depth':[1, 2, 3, 4, 5, 6, 7, None]}]
rfc_gs = GridSearchCV(estimator=rfc, param_grid=rfc_grid, cv=10, scoring='f1', n_jobs=-1)
rfc_gs.fit(over_x, over_y)
# rfc_f1_scores = cross_val_score(estimator=rfc_gs, X=over_x, y=over_y, scoring='f1', cv=5)
# print('CV clf f1 score:%.3f +/- %.3f' % (np.mean(rfc_f1_scores), np.std(rfc_f1_scores)))

print(rfc_gs.best_params_)
# {'bootstrap': False, 'max_depth': None, 'n_estimators': 40}
print(rfc_gs.best_score_)
# 0.9836533023676681

#%%

# 测试精度train
test_x = train_data_.drop('target', axis=1)
test_y = train_data_['target']

y_pred = rfc_gs.predict(test_x)
print("TRAIN")
print(classification_report(test_y, y_pred))

# 测试精度test
test_x = test_data_.drop('target', axis=1)
test_y = test_data_['target']

y_pred = rfc_gs.predict(test_x)
print("TEST")
print(classification_report(test_y, y_pred))

#%%

train_data_.plot.scatter(x=0,
                      y=1,
                      c='target',
                      colormap='viridis')

#%%
from sklearn.decomposition import PCA

X = train_data_.drop('target', axis=1)
X

#%%
pca = PCA(n_components=10)
pca.fit(X)

print (pca.explained_variance_ratio_)
print (pca.explained_variance_)


#%%

X_new = pca.transform(X)
plt.scatter(X_new[:, 0], X_new[:, 1],marker='o',c=train_data_['target'])
plt.show()

#%%
new_df = df_test_under.sample(frac=1, random_state=42)
new_df.head()

# %%

X = new_df.drop('target', axis=1)
y = new_df['target']

y

#%%
train_data_.iloc[:, 6].value_counts()

# %%

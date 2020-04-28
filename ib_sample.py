from collections import Counter

import pandas as pd
from imblearn.over_sampling import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split

train_data_ = pd.read_csv('dataset/train.data', sep='\s+', header=None)
test_data_ = pd.read_csv('dataset/test.data', sep='\s+', header=None)

train_data_.rename(columns={10: 'target'}, inplace=True)
test_data_.rename(columns={10: 'target'}, inplace=True)


def split_xy(data):
    _x = data.drop('target', axis=1)
    _y = data['target']
    return _x, _y


X, y = split_xy(train_data_)
print('Original dataset shape %s' % Counter(y))
# Original dataset shape Counter({1: 900, 0: 100})

# 三种不同的超采样方式
# sm = ADASYN(random_state=42)
# sm = SMOTE(random_state=42)
sm = BorderlineSMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
print('Resampled dataset shape %s' % Counter(y_res))
# Resampled dataset shape Counter({0: 900, 1: 900})


# RF

# over_x, over_y = split_xy(df_test_over)
over_x, over_y = X_res, y_res

# X_train, X_test, y_train, y_test = train_test_split(over_x, over_y, test_size=0.2, random_state=42)

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

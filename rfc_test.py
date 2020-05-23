from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

from util import *

# 二分类，逻辑回归
print(train_data_)

X, y = split_xy(train_data_)

# 分割数据集 划分为训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# 进行标准化处理   因为目标结果经过sigmoid函数转换成了[0,1]之间的概率，所以目标值不需要进行标准化。
# std = StandardScaler()
# x_train = std.fit_transform(x_train)
# x_test = std.transform(x_test)

# 预测
clf = RandomForestClassifier(n_estimators=100)
clf.fit(x_train, y_train)
print_report(clf, x_test, y_test, "pre-train", "RandomForest")

# # GridSearchCV 搜索最优 knn ------------------------------------------------------------------------------
param_grid = {'bootstrap': [False, True], 'n_estimators': [3, 4, 5, 6, 7, 8, 9, 10, 15, 40, 50, 60, 65, 70, 90, 100],
              'max_depth': [1, 3, 5, 7]}
param_grid = {'bootstrap': [True], 'n_estimators': [100, 120, 150, 170, 200],
              'max_depth': [1, 3, 5, 7]}
clf = RandomForestClassifier(n_jobs=-1, class_weight='balanced')
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=10, scoring='f1_macro', n_jobs=-1)
grid_search.fit(x_train, y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)
print(grid_search.best_estimator_)
# means = grid_search.cv_results_['mean_test_score']
# params = grid_search.cv_results_['params']
# for mean, param in zip(means, params):
#     print("%f  with:   %r" % (mean, param))

print_report(grid_search.best_estimator_, x_test, y_test, "best-train", "RandomForest")

x_test, y_test = split_xy(test_data_)
print_report(grid_search.best_estimator_, x_test, y_test, "best-test", "RandomForest")
plot_mat(grid_search.best_estimator_, x_test, y_test, "RandomForest")
print_latex(grid_search.best_estimator_, x_test, y_test, "RandomForest")

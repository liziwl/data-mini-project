from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC, LinearSVC

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

# 逻辑回归预测
# kernel = 'linear','poly','rbf'
# gamma = 'auto', 'scale'
svm = SVC(C=0.01, kernel='rbf', class_weight='balanced', gamma='auto')
svm.fit(x_train, y_train)

print_report(svm, x_test, y_test, "train", "svm")

x_test, y_test = split_xy(test_data_)
print_report(svm, x_test, y_test, "test", "svm")

# # GridSearchCV 搜索最优 SVC ------------------------------------------------------------------------------
# param_grid = {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000], "kernel": ['rbf'],
#               'gamma': ['auto']}
# clf = SVC(class_weight='balanced')
# grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
# grid_search.fit(x_train, y_train)
# print(grid_search.best_params_)
# print(grid_search.best_score_)
# print(grid_search.best_estimator_)
# means = grid_search.cv_results_['mean_test_score']
# params = grid_search.cv_results_['params']
# for mean, param in zip(means, params):
#     print("%f  with:   %r" % (mean, param))


# ------------------------------------------------------------
svm = SVC(C=0.01, kernel='rbf', class_weight='balanced', gamma='auto')
svm.fit(x_train, y_train)

print_report(svm, x_test, y_test, "train", "svm-best", )

x_test, y_test = split_xy(test_data_)
print_report(svm, x_test, y_test, "test", "svm-best")

plot_mat(svm, x_test, y_test, "svm")
print_latex(svm, x_test, y_test, "svm")

# # GridSearchCV 搜索最优 LinearSVC ------------------------------------------------------------------------------
# param_grid = {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
# clf = LinearSVC(class_weight='balanced')
# grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
# grid_search.fit(x_train, y_train)
# print(grid_search.best_params_)
# print(grid_search.best_score_)
# print(grid_search.best_estimator_)
# means = grid_search.cv_results_['mean_test_score']
# params = grid_search.cv_results_['params']
# for mean, param in zip(means, params):
#     print("%f  with:   %r" % (mean, param))

# ------------------------------------------------------------
svm = LinearSVC(C=10, class_weight='balanced')
svm.fit(x_train, y_train)

print_report(svm, x_test, y_test, "train", "LinearSVC-best", )

x_test, y_test = split_xy(test_data_)
print_report(svm, x_test, y_test, "test", "LinearSVC-best")

plot_mat(svm, x_test, y_test, "LinearSVC")
print_latex(svm, x_test, y_test, "LinearSVC")

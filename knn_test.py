from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC, LinearSVC

from util import *
from sklearn.neighbors import KNeighborsClassifier

# 二分类，逻辑回归
print(train_data_)

X, y = split_xy(train_data_)
X, y = resample(X, y)

# 分割数据集 划分为训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# 进行标准化处理   因为目标结果经过sigmoid函数转换成了[0,1]之间的概率，所以目标值不需要进行标准化。
# std = StandardScaler()
# x_train = std.fit_transform(x_train)
# x_test = std.transform(x_test)

# 预测
# clf = KNeighborsClassifier(n_neighbors=3, weights='distance', n_jobs=-1)
# clf.fit(x_train, y_train)
# print_report(clf, x_train, y_train, "pre-train", "KNN")

# # GridSearchCV 搜索最优 knn ------------------------------------------------------------------------------
param_grid = {"n_neighbors": [3, 5, 7, 9], "weights": ['uniform', 'distance'],
              'p': [1, 2]}
clf = KNeighborsClassifier(n_jobs=-1)
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
grid_search.fit(x_train, y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)
print(grid_search.best_estimator_)
# means = grid_search.cv_results_['mean_test_score']
# params = grid_search.cv_results_['params']
# for mean, param in zip(means, params):
#     print("%f  with:   %r" % (mean, param))

print_report(grid_search.best_estimator_, x_test, y_test, "best-train", "KNN")

x_test, y_test = split_xy(test_data_)
print_report(grid_search.best_estimator_, x_test, y_test, "best-test", "KNN")
plot_mat(grid_search.best_estimator_, x_test, y_test, "KNN-re")
print_latex(grid_search.best_estimator_, x_test, y_test, "KNN-re")

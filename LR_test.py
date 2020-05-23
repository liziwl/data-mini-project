from sklearn.linear_model import LogisticRegression
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

# 逻辑回归预测
lg = LogisticRegression(C=1.0, class_weight='balanced', solver='liblinear',
                        n_jobs=-1)  # 默认使用L2正则化避免过拟合，C=1.0表示正则力度(超参数，可以调参调优)
lg.fit(x_train, y_train)

# 回归系数
print(lg.coef_)
# 进行预测
print_report(lg, x_test, y_test, "train", "lg", )

x_test, y_test = split_xy(test_data_)
print_report(lg, x_test, y_test, "test", "lg")

# # GridSearchCV 搜索最优 ------------------------------------------------------------------------------
# param_grid = {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000], "penalty": ['l1', 'l2']}
# lg = LogisticRegression(class_weight='balanced', solver='liblinear',
#                         n_jobs=-1)
# grid_search = GridSearchCV(lg, param_grid=param_grid, cv=5, scoring='f1_macro')
# grid_search.fit(x_train, y_train)
# print(grid_search.best_params_)
# print(grid_search.best_score_)
# print(grid_search.best_estimator_)
# means = grid_search.cv_results_['mean_test_score']
# params = grid_search.cv_results_['params']
# for mean, param in zip(means, params):
#     print("%f  with:   %r" % (mean, param))

# 最优结果
"""{'C': 0.01, 'penalty': 'l1'}
0.5486103096590262
LogisticRegression(C=0.01, class_weight='balanced', dual=False,
                   fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                   max_iter=100, multi_class='auto', n_jobs=-1, penalty='l1',
                   random_state=None, solver='liblinear', tol=0.0001, verbose=0,
                   warm_start=False)"""

# -------------------------------------------------------------------
lg = LogisticRegression(class_weight='balanced', solver='liblinear',
                        n_jobs=-1, C=0.01, penalty='l1')
lg.fit(x_train, y_train)
print_report(lg, x_test, y_test, "train", "lg-best", )

x_test, y_test = split_xy(test_data_)
print_report(lg, x_test, y_test, "test", "lg-best")
print_latex(lg, x_test, y_test)
"""
--------------------train@lg-best--------------------
              precision    recall  f1-score   support

          -1       0.95      0.99      0.97      1955
           1       0.26      0.08      0.12       117

    accuracy                           0.94      2072
   macro avg       0.61      0.53      0.54      2072
weighted avg       0.91      0.94      0.92      2072

--------------------test@lg-best--------------------
              precision    recall  f1-score   support

          -1       0.95      0.99      0.97      1955
           1       0.26      0.08      0.12       117

    accuracy                           0.94      2072
   macro avg       0.61      0.53      0.54      2072
weighted avg       0.91      0.94      0.92      2072
"""


# ------------------------------------------------------------
titles_options = [("Confusion matrix, without normalization","confus_mat.png", None),
                  ("Normalized confusion matrix","confus_mat-norm.png", 'true')]
for title, file_name, normalize in titles_options:
    disp = plot_confusion_matrix(lg, x_test, y_test,
                                 display_labels=[-1,1],
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)
    plt.savefig(f'lg_{file_name}', bbox_inches='tight', dpi=400, pad_inches=0.05)


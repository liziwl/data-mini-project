import pandas as pd
from sklearn.metrics import classification_report, plot_confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
import matplotlib.pyplot as plt


train_data_ = pd.read_csv('dataset/train.data', sep='\s+', header=None)
test_data_ = pd.read_csv('dataset/test.data', sep='\s+', header=None)

train_data_.rename(columns={10: 'target'}, inplace=True)
test_data_.rename(columns={10: 'target'}, inplace=True)


def split_xy(data):
    _x = data.drop('target', axis=1)
    _y = data['target']
    return _x, _y


def print_report(clf, x_test, y_real, data_name=None, clf_name=None):
    if data_name is None:
        data_name = " " * 10
    if clf_name is None:
        data_name = " " * 10
    print("-" * 20 + f"{data_name}@{clf_name}" + "-" * 20)
    y_pred = clf.predict(x_test)
    print(classification_report(y_real, y_pred))


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

# # GridSearchCV 搜索最优 ------------------------------------------------------------------------------
param_grid = {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000], "kernel": ['rbf'],
              'gamma': ['auto']}
clf = SVC(class_weight='balanced')
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
grid_search.fit(x_train, y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)
print(grid_search.best_estimator_)
means = grid_search.cv_results_['mean_test_score']
params = grid_search.cv_results_['params']
for mean, param in zip(means, params):
    print("%f  with:   %r" % (mean, param))


# ------------------------------------------------------------
svm = SVC(C=0.01, kernel='rbf', class_weight='balanced', gamma='auto')
svm.fit(x_train, y_train)

svm.fit(x_train, y_train)
print_report(svm, x_test, y_test, "train", "svm-best", )

x_test, y_test = split_xy(test_data_)
print_report(svm, x_test, y_test, "test", "svm-best")

titles_options = [("Confusion matrix, without normalization","confus_mat.png", None),
                  ("Normalized confusion matrix","confus_mat-norm.png", 'true')]
for title, file_name, normalize in titles_options:
    disp = plot_confusion_matrix(svm, x_test, y_test,
                                 display_labels=[-1,1],
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)
    plt.savefig(f'svm_{file_name}', bbox_inches='tight', dpi=400, pad_inches=0.05)


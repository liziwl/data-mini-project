import pandas as pd
from sklearn.metrics import classification_report, plot_confusion_matrix, recall_score, precision_score, accuracy_score, f1_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import *
import numpy as np
from sklearn.utils import class_weight

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


def print_latex(clf, x_test, y_real, caption):
    y_pred = clf.predict(x_test)
    p0 = precision_score(y_real, y_pred, average=None, labels=[-1, 1])
    p_a = precision_score(y_real, y_pred, average='macro', labels=[-1, 1])

    r0 = recall_score(y_real, y_pred, average=None, labels=[-1, 1])
    r_a = recall_score(y_real, y_pred, average='macro', labels=[-1, 1])

    f1_0 = f1_score(y_real, y_pred, average=None, labels=[-1, 1])
    f1_a = f1_score(y_real, y_pred, average='macro', labels=[-1, 1])

    str1 = f'''\\begin{{table}}[!h]
    \\centering
    \\renewcommand{{\\arraystretch}}{{1.5}}
    \\begin{{tabular}}{{|r|c|c|c|}}
        \\hline
                  & Precision & Recall & F1-score \\\\ \\hline
        -1        &     {p0[0]:.5f}     &     {r0[0]:.5f}   &    {f1_0[0]:.5f}      \\\\ \\hline
        1         &     {p0[1]:.5f}      &     {r0[1]:.5f}   &   {f1_0[1]:.5f}       \\\\ \\hline
        macro avg &      {p_a:.5f}     &     {r_a:.5f}   &    {f1_a:.5f}      \\\\ \\hline
    \end{{tabular}}
    \caption{{{caption}}}
\end{{table}}'''
    print(str1)
    with open(f"{caption}.tex", 'w', encoding='utf8') as f:
        print(clf, file=f)
        print(str1, file=f)


def plot_mat(clf, x_test, y_test, file_prefix):
    titles_options = [("Confusion matrix, without normalization", "confus_mat.png", None),
                      ("Normalized confusion matrix", "confus_mat-norm.png", 'true')]
    for title, file_name, normalize in titles_options:
        disp = plot_confusion_matrix(clf, x_test, y_test,
                                     display_labels=[-1, 1],
                                     cmap=plt.cm.Blues,
                                     normalize=normalize)
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)
        plt.savefig(f'{file_prefix}_{file_name}', bbox_inches='tight', dpi=400, pad_inches=0.05)


def resample(X, y):
    sm = SMOTE(random_state=42, sampling_strategy='minority')
    # sm = ADASYN(random_state=42)
    # sm = SMOTE(random_state=42)
    _X, _y = sm.fit_resample(X, y)
    return _X, _y


def cls_weight(y_true):
    class_weights = list(class_weight.compute_class_weight('balanced', np.unique(y_true), y_true))
    w_array = np.ones(y_true.shape[0], dtype='float')
    for i, val in enumerate(y_true):
        w_array[i] = class_weights[val-1]
    return w_array


if __name__ == "__main__":
    x, y = split_xy(train_data_)
    print(cls_weight(y))

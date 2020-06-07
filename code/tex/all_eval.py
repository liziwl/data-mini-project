# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import re
import os

file_list = []
for fn in os.listdir("."):
    if fn.endswith("tex"):
        file_list.append(fn)

file_list.sort()

# %%

data = dict()
for fn in file_list:
    with open(fn, 'r', encoding='utf8') as f:
        lines = f.readlines()[6:9]
        tmp = []
        for it in lines:
            it = it.strip()
            it = it.replace("\\\\ \\hline", "")
            it = it.split("&")
            tmp.append([j.strip() for j in it])
        data[fn] = tmp


# %%
data

# %%

class_ = [(0, "-1", "neg.png"),
          (1, "1", "pos.png"),
          (2, "Macro Avg", "avg.png")]
for n_class, ax_title, png_path in class_:
    labels = file_list
    precision_v = [float(data[f][n_class][1]) for f in file_list]
    recall_v = [float(data[f][n_class][2]) for f in file_list]
    f1_v = [float(data[f][n_class][3]) for f in file_list]

    show_label = [i.replace(".tex", "")[2:] for i in labels]
    # ['LR', 'LinearSVC', 'svm', 'KNN-re', 'RandomForest', 'AdaBoost-re', 'GradientBoosting-re', 'LightGBM-RE', 'xgboost']
    show_label = ['LR', 'LinearSVC', 'SVM', 'KNN', 'RF',
                  'AdaBoost', 'GBM', 'Light\nGBM', 'XGBoost']

    print(list(zip(labels, show_label)))
    # print(show_label)
    # print(precision_v)
    # print(recall_v)
    # print(f1_v)

    margin = 0.1
    num_items = 3
    width = (1.-2.*margin)/num_items
    x = np.arange(len(labels))  # the label locations

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, precision_v, width, label='Precision')
    rects2 = ax.bar(x, recall_v, width, label='Recall')
    rects3 = ax.bar(x + width, f1_v, width, label='F1')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title(f'Scores of Class {ax_title}')
    ax.set_xticks(x)
    ax.set_xticklabels(show_label)
    ax.set_ylim(0, 1.5)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            # ax.annotate(f'{np.format_float_positional(height, precision=2)[2:]}',
            #             xy=(rect.get_x() + rect.get_width() / 2, height),
            #             xytext=(0, 3),  # 3 points vertical offset
            #             textcoords="offset points",
            #             ha='center', va='bottom', fontsize=9)
            ax.annotate(f'{int(100*height%100)}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    fig.tight_layout()

    plt.savefig(f'{png_path}', bbox_inches='tight', dpi=400, pad_inches=0.05)
    plt.show()

# %%

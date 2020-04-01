import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

train_data_ = pd.read_csv('dataset/train.data', sep='\s+', header=None)
test_data_ = pd.read_csv('dataset/test.data', sep='\s+', header=None)


# train_data_.iloc[:,-1].value_counts()
# -1    7819
#  1     466
# Name: 10, dtype: int64
# -1 0.9437537718768859
# 1 0.05624622812311406
# 所以要考虑数据的分布


def acu_curve(y, prob):
    fpr, tpr, threshold = roc_curve(y, prob)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值

    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")

    plt.show()


train_data = train_data_.iloc[:, :-1]
train_target = train_data_.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(train_data, train_target, test_size=0.4, random_state=42)

rf = RandomForestClassifier(oob_score=True, random_state=42)
rf.fit(X_train, y_train)
print(rf.oob_score_)

_prob_y = rf.predict(X_test)
acu_curve(y_test, _prob_y)

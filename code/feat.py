# %%
# 特征分析 相关系数
import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt

train_data_ = pd.read_csv('dataset/train.data', sep='\s+', header=None)
test_data_ = pd.read_csv('dataset/test.data', sep='\s+', header=None)

train_data_.rename(columns={10: 'target'}, inplace=True)
test_data_.rename(columns={10: 'target'}, inplace=True)


def split_xy(data):
    _x = data.drop('target', axis=1)
    _y = data['target']
    return _x, _y


_x, _y = split_xy(train_data_)

# plt.figure(figsize=(10, 10))
# sns.heatmap(_x.corr(), annot=True)
# plt.savefig('train_corr.png', bbox_inches='tight', dpi=400, pad_inches=0.05)
# plt.show()

# %%


from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="white")

d = train_data_

# Compute the correlation matrix
corr = d.corr(method='pearson')
# method = {‘pearson’,‘kendall’,‘spearman’}

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, s=70, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.savefig('train_corr2.png', bbox_inches='tight', dpi=400, pad_inches=0.05)
plt.show()

# %%
# 特征分析PCA

from sklearn.decomposition import PCA

X, Y = split_xy(train_data_)
pca = PCA(n_components=10)
pca.fit(X)

print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
plt.bar(range(10), pca.explained_variance_ratio_)
plt.title("explained_variance_ratio_")
plt.show()
# plt.bar(range(10),pca.explained_variance_)
# plt.title("explained_variance_")
# plt.show()

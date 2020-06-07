# %%

from sklearn.utils.class_weight import compute_class_weight

class_weight = 'balanced'
label = [0] * 9 + [1]*1 + [2, 2]
print(label)  # [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2]
classes = [0, 1, 2]
weight = compute_class_weight(class_weight, classes, label)
print(weight)  # [ 0.44444444 4. 2. ]
print(.44444444 * 9)  # 3.99999996
print(4 * 1)  # 4
print(2 * 2)  # 4

# %%

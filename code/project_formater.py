from util import *
import csv
X, y = split_xy(train_data_)

path = 'a.tsv'
X.to_csv(path, sep="\t", quoting=csv.QUOTE_NONE, header=0, index=0)

path = 'b.tsv'
y.to_csv(path, sep="\t", quoting=csv.QUOTE_NONE, header=0, index=0)
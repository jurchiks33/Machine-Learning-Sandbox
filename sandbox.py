import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 
from sklearn.preprocessing import scale
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.cm as cm

digits = load_digits()
data = digits.data

n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels = digits.target


pca = PCA(n_components=10)
data_r = pca.fit(data).transform(data)

print('explain variance ratio (first two components) : %s' %
      str(pca.explained_variance_ratio_))
print('explain variance ratio (first two components) : %s' %
      str(sum(pca.explained_variance_ratio_)))
# import numpy as np
# from sklearn.datasets import load_digits
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA 
# from sklearn.preprocessing import scale
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# import matplotlib.cm as cm

# digits = load_digits()
# data = digits.data

# n_samples, n_features = data.shape
# n_digits = len(np.unique(digits.target))
# labels = digits.target


# pca = PCA(n_components=10)
# data_r = pca.fit(data).transform(data)

# print('explain variance ratio (first two components) : %s' %
#       str(pca.explained_variance_ratio_))
# print('explain variance ratio (first two components) : %s' %
#       str(sum(pca.explained_variance_ratio_)))



# x = np.arange(10)
# ys = [i+x+(i*x)**2 for i in range(10)]

# plt.figure()
# colors = cm.rainbow(np.linspace(0, 1, len(ys)))
# for c, i, target_name in zip(colors, [1,2,3,4,5,6,7,8,9,10], labels):
#     plt.scatter(data_r[labels == i, 0], data_r[labels == i, 1],
#     c=c, alpha= 0.4)
#     plt.legend()
#     plt.title('Scatterplot of Points plotted in first \n'
#               '10 Principal Components')
#     plt.show()

# plt.figure()
# colors = cm.rainbow(np.linspace(0, 1, len(np.unique(labels))))
# target_names = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5',
#                 'Class 6', 'Class 7', 'Class 8', 'Class 9', 'Class 10']

# for c, i, target_name in zip(colors, range(1, 11), target_names):
#     plt.scatter(data_r[labels == i, 0], data_r[labels == i, 1], c=[c], alpha=0.4, label=target_name)

# plt.legend()
# plt.title('Scatterplot of Points plotted in first \n10 Principal Components')
# plt.show()


from time import time
import numpy as np
import matplotlib.pyplot as plt

np.random.seed()


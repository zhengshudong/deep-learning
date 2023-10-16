# 不要因为走得太远，就忘记了最初为何而出发
# 旅行不只是为了见证鲜花盛放，诗歌、城邦、飞鸟，这些都是你前行的意义
# 在抵达那一切之前，用你的心，去见证这个世界吧
# 开发时间：2023/9/23 10:12

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# 加载CSV表格数据
data = pd.read_csv("../数据集/vessel_clustering.csv")

# 数据预处理和降维
X = data.iloc[:, :10]  # 选择前十列作为特征
X_normalized = (X - X.mean()) / X.std()  # 数据标准化
pca = PCA(n_components=3)  # 设置PCA降维到三维
X_pca = pca.fit_transform(X_normalized)

# 使用K-means进行聚类
n_clusters = 5  # 聚类数目
kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
kmeans.fit(X_pca)
labels = kmeans.labels_

# 可视化聚类结果
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=labels)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.show()

# 输出每个聚类的主导特征
for i in range(n_clusters):
    cluster_data = X.iloc[labels == i, :]
    dominant_features = cluster_data.mean(axis=0).sort_values(ascending=False)
    print(f"Cluster {i+1} dominant features:")
    print(dominant_features[:3])  # 替换为你想要输出的特征数量
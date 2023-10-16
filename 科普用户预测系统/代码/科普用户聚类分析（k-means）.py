# 不要因为走得太远，就忘记了最初为何而出发
# 旅行不只是为了见证鲜花盛放，诗歌、城邦、飞鸟，这些都是你前行的意义
# 在抵达那一切之前，用你的心，去见证这个世界吧
# 开发时间：2023/9/23 9:54

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 读取CSV文件
data = pd.read_csv("../数据集/vessel_k_means.csv")

# 提取特征数据
features = data.iloc[:, :15].values

# 使用K-means算法进行聚类
kmeans = KMeans(n_clusters=5, n_init='auto')  # 设置聚类簇数
kmeans.fit(features)
labels = kmeans.labels_

# 输出聚类结果图
plt.scatter(features[:, 0], features[:, 1], c=labels)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Clustering Result')
plt.show()

# 输出每一种聚类的主导特征
centroids = kmeans.cluster_centers_
for i in range(len(centroids)):
    dominant_features = pd.Series(centroids[i])
    print('Cluster', i+1, 'Dominant Features:')
    print(dominant_features.nlargest(3))  # 输出前3个主导特征
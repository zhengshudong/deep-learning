# 不要因为走得太远，就忘记了最初为何而出发
# 旅行不只是为了见证鲜花盛放，诗歌、城邦、飞鸟，这些都是你前行的意义
# 在抵达那一切之前，用你的心，去见证这个世界吧
# 开发时间：2023/9/24 11:54

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 读取数据
data = pd.read_csv('../数据集/vessel_classification.csv')

# 提取特征和标签
X = data.iloc[:, :10].values    # 特征列
y = data.iloc[:, 10:].values     # 标签列

# 编码标签（如果标签是字符串）
le = LabelEncoder()
y = le.fit_transform(y.ravel())

# 特征缩放
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 构建SVM模型
model = SVC(kernel='poly',probability=True)
model.fit(X_train, y_train.ravel())

# 预测概率
y_pred_prob = model.predict_proba(X_test)
y_pred_prob = np.argmax(y_pred_prob, axis=1)

# 计算均方误差
mse = mean_squared_error(y_test, y_pred_prob)
print('Mean Squared Error:', mse)

# 计算ROC曲线和AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# 绘制ROC曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# 支持向量可视化
def plot_support_vectors(model, X, y):
    sv = model.support_vectors_
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.scatter(sv[:, 0], sv[:, 1], c='r', marker='x', label='Support Vectors')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Support Vectors')
    plt.legend()
    plt.show()

plot_support_vectors(model, X_scaled[:, :2], y)

# 样本权重可视化
def plot_sample_weights(model, X, y):
    weights = np.abs(model.support_vectors_)[0]
    plt.bar(range(X.shape[1]), weights)
    plt.xticks(range(X.shape[1]), range(1, X.shape[1]+1))
    plt.xlabel('Features')
    plt.ylabel('Weight')
    plt.title('Sample Weights')
    plt.show()

plot_sample_weights(model, X_scaled, y)

# 特征重要性可视化
def plot_feature_importance(model, X, y):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.bar(range(X_pca.shape[1]), pca.explained_variance_ratio_)
    plt.xticks(range(X_pca.shape[1]), range(1, X_pca.shape[1]+1))
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Feature Importance')
    plt.show()

plot_feature_importance(model, X_scaled, y)

# 计算精度
accuracy = accuracy_score(y_test, y_pred_prob)
print(accuracy)
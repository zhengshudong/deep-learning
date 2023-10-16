# 不要因为走得太远，就忘记了最初为何而出发
# 旅行不只是为了见证鲜花盛放，诗歌、城邦、飞鸟，这些都是你前行的意义
# 在抵达那一切之前，用你的心，去见证这个世界吧
# 开发时间：2023/9/25 14:31

import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tensorflow as tf

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

# 构建贝叶斯分类器
model = BernoulliNB()

# 训练分类器
model.fit(X_train, y_train)

# 预测概率
y_pred = model.predict_proba(X_test)

# 计算ROC曲线的假正例率（FPR）和真正例率（TPR）
fpr, tpr, thresholds = roc_curve(y_test, y_pred[:, 1], pos_label=1)

# 计算AUC（Area Under the Curve）
auc = roc_auc_score(y_test, y_pred[:, 1], multi_class='ovo')

# 绘制ROC曲线
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# 计算混淆矩阵
yy_pred = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_test, yy_pred)
print("Confusion    matrix:\n",  cm)
print("Classification    report:\n",  classification_report(y_test,  yy_pred))

# 特征重要性可视化
feature_importance = model.theta_[1] - model.theta_[0]
plt.bar(range(len(feature_importance)), feature_importance)
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.show()

# 特征空间投影可视化
pca = PCA(n_components=2)
X_transformed = pca.fit_transform(X)
plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y, edgecolors='k')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Feature Space Projection')
plt.show()

# 计算精度
accuracy = accuracy_score(y_test, yy_pred)
print(accuracy)

# 不要因为走得太远，就忘记了最初为何而出发
# 旅行不只是为了见证鲜花盛放，诗歌、城邦、飞鸟，这些都是你前行的意义
# 开发时间：2023/5/18  18：05


#  导入所需模块和数据
import  pandas  as  pd
import numpy as np
import  matplotlib.pyplot  as  plt
from  sklearn.ensemble  import  RandomForestClassifier
from  sklearn.model_selection  import  GridSearchCV,  train_test_split,  StratifiedKFold,  cross_val_score
from  sklearn.metrics  import  accuracy_score,  confusion_matrix,  classification_report,  roc_curve
from  sklearn.manifold  import  MDS
from  sklearn.pipeline  import  Pipeline
from  sklearn.compose  import  ColumnTransformer
from  sklearn.preprocessing  import  StandardScaler,  OneHotEncoder
from  sklearn.metrics  import  auc, accuracy_score, RocCurveDisplay
from sklearn import metrics

#  忽略版本警告
import warnings
warnings.filterwarnings('ignore', message='The default value of `normalized_stress` will change to')
normalized_stress = 'auto'

#  读取数据并赋值
df  =  pd.read_csv("../数据集/vessel_classification.csv")
X  =  df.drop("target",  axis=1)
y  =  df["target"]

#  设置参数列表
param_grid  =  {
        "n_estimators":  [100, 150, 200, 250],
        "criterion":  ["gini",  "entropy"],
        "max_depth":  [5, 10, 15],
        "min_samples_split":  [2, 4, 6],
        "min_samples_leaf":  [1, 2, 3],
        "max_features":  [3, 4, 5]
}

#  采用GridSearchCV方法自动搜索最优参数
rf  =  RandomForestClassifier(random_state=42,  n_jobs=-1)
grid_search  =  GridSearchCV(rf,  param_grid,  scoring="accuracy",  cv=10,  n_jobs=-1)
grid_search.fit(X,  y)
best_model  =  grid_search.best_estimator_

#  输出最优参数和模型性能
print(f"Best    parameters:    {grid_search.best_params_}")
print(f"Best    accuracy:    {grid_search.best_score_}")
y_pred_train  =  best_model.predict(X)
print("Accuracy    on    training    set:",  accuracy_score(y,  y_pred_train))
cv  =  StratifiedKFold(n_splits=5,  shuffle=True,  random_state=42)
cv_scores  =  cross_val_score(best_model,  X,  y,  cv=cv,  scoring='accuracy')
print("Cross-validation    accuracy:",  cv_scores.mean())
RocCurveDisplay.from_estimator(best_model,  X,  y)
plt.title("ROC    Curve    of    Best    Model")
plt.legend()
plt.show()

#  输出特征重要性
importances  =  best_model.feature_importances_
indices  =  pd.Series(importances,  index=X.columns).sort_values()
plt.barh(indices.index,  indices)
plt.title("Feature    Importances")
plt.xlabel("Relative        Importance")
plt.ylabel("Features")
plt.show()

#  划分训练集和测试集
X_train,  X_test,  y_train,  y_test  =  train_test_split(X,  y,  test_size=0.2,  random_state=42)

#  绘制ROC曲线
y_prob  =  best_model.predict_proba(X_test)
y_prob = np.argmax(y_prob, axis=1)
fpr,  tpr,  _  =  roc_curve(y_test,  y_prob)
roc_auc  =  auc(fpr,  tpr)
plt.plot(fpr,  tpr,  label=f"AUC    =    {roc_auc:.2f}")
plt.plot([0,  1],  [0,  1],  linestyle="--")
plt.xlabel("False    Positive    Rate")
plt.ylabel("True    Positive    Rate")
plt.title("ROC    Curve")
plt.legend()
plt.show()

#  混淆矩阵
y_pred  =  best_model.predict(X_test)
print("Confusion    matrix:\n",  confusion_matrix(y_test,  y_pred))
print("Classification    report:\n",  classification_report(y_test,  y_pred))

#  使用MDS方法将特征降至2维，并绘制分类结果图
mds  =  MDS(n_components=2,  random_state=42)
X_2d  =  mds.fit_transform(X)
plt.figure(figsize=(8,  6))
plt.scatter(X_2d[:,  0],  X_2d[:,  1],  c=y,  cmap="tab10")
plt.title("2D    Classification    Plot    of    Vessel    Health    Prediction")
plt.show()

# 计算精度

accuracy = accuracy_score(y_test, y_prob)
print(accuracy)
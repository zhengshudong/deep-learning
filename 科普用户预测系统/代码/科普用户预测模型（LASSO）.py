import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge# 导入岭回归算法
from sklearn.linear_model import Lasso# 导入lasso回归算法
from sklearn.linear_model import LassoLars
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score, accuracy_score, roc_curve, auc
dataset=pd.read_csv("../数据集/vessel_multi_classification.csv")
data_iter=dataset.dropna().reset_index()
del data_iter["index"]

train=data_iter.drop(["target"],axis=1)
X_train,X_test,y_train,y_test=train_test_split(train,data_iter["target"],random_state=23)

lasso=Lasso(alpha=0.3,max_iter=15)
lasso.fit(X_train,y_train)
print("Lasso训练模型得分:"+str(r2_score(y_train,lasso.predict(X_train))))#训练集
print("Lasso待测模型得分:"+str(r2_score(y_test,lasso.predict(X_test))))#待测集
print(accuracy_score(y_test, np.around(lasso.predict(X_test),0).astype(int)))

lasso2 = LassoLars(alpha = 0.3,max_iter=15)
lasso2.fit(X_train,y_train)
print("LassoLars训练模型得分:"+str(r2_score(y_train,lasso2.predict(X_train))))#训练集
print("LassoLars待测模型得分:"+str(r2_score(y_test,lasso2.predict(X_test))))#待测集
print(accuracy_score(y_test, np.around(lasso2.predict(X_test),0).astype(int)))

lasso3 = LassoCV(alphas=[0.3,0.4,0.5,0.1],max_iter=15,cv = 7).fit(X_train,y_train)
print(lasso3.alpha_)
print("LassoCV训练模型得分:"+str(r2_score(y_train,lasso3.predict(X_train))))#训练集
print("LassoCV待测模型得分:"+str(r2_score(y_test,lasso3.predict(X_test))))#待测集
print(accuracy_score(y_test, np.around(lasso3.predict(X_test),0).astype(int)))

coef = pd.Series(lasso3.coef_, index = dataset.columns[:-1])
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

imp_coef = pd.concat([coef.sort_values().head(5),
                     coef.sort_values().tail(5)])

matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
ax=coef.plot(kind = "barh")

plt.title("Coefficients in the Lasso Model")
plt.show()
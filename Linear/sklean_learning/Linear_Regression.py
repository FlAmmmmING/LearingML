import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# 如何用最小二乘法计算theta?

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# 数据构造
X = 2 * np.random.rand(100, 1)
# 随机抖动
y = 4 + 3 * X + np.random.rand(100, 1)

# 按照公式求theta
X_b = np.c_[np.ones((100, 1)), X]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
Y_predict = X_new_b.dot(theta_best)
print(theta_best)
plt.plot(X_new, Y_predict, 'r--')
plt.plot(X, y, 'b.')
plt.xlabel('x1')
plt.ylabel('y')
plt.axis([0, 2, 0, 15])
# plt.show()

# 用API写
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)
# 权重参数与偏置参数
print(lin_reg.coef_, lin_reg.intercept_)

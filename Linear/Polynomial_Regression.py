import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X ** 2 + np.random.randn(m, 1)
# plt.plot(X, y, 'b.')
# plt.xlabel('X_1')
# plt.ylabel('y')
# plt.axis([-3, 3, -5, 10])
# plt.show()

poly = PolynomialFeatures(degree=10, include_bias=False)
poly_X = poly.fit_transform(X)
lin_reg = LinearRegression()
lin_reg.fit(poly_X, y)
X_new = np.linspace(-3, 3, 100).reshape(100, 1)
X_new_poly = poly.transform(X_new)
y_new = lin_reg.predict(X_new_poly)
plt.plot(X, y, 'b.')
plt.plot(X_new, y_new, 'r--', label='predicted')
plt.axis([-3, 3, -5, 10])
plt.legend()
plt.show()
from sklearn.linear_model import Ridge
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from matplotlib import pyplot as plt
from sklearn.linear_model import Lasso
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)
m = 20
X = 3 * np.random.rand(m, 1)
y = 0.5 * X + np.random.randn(m, 1) / 1.5 + 1
X_new = np.linspace(0, 3, 100).reshape(100, 1)


def plot_model(model_class, polynomial, alphas, **model_kargs):
    for alpha, style in zip(alphas, ('b-', 'g--', 'r:')):
        model = model_class(alpha, **model_kargs)
        if polynomial:
            model = Pipeline([('polynomial', PolynomialFeatures(degree=10, include_bias=False)),
                              ('StandardScaler', StandardScaler()),
                              ('lin_reg', model)])
        model.fit(X, y)
        y_new_reg = model.predict(X_new)
        lw = 2 if alpha > 0 else 1
        plt.plot(X_new, y_new_reg, style, linewidth=lw, label='alpha={}'.format(alpha))
    plt.plot(X, y, 'b.', linewidth=3)
    plt.legend()


plt.figure(figsize=(10, 5))
plt.subplot(221)
plot_model(Ridge, polynomial=False, alphas=(0, 0.1, 1))
plt.subplot(222)
plot_model(Ridge, polynomial=True, alphas=(0, 10 ** -1, 1))
plt.subplot(223)
plot_model(Lasso, polynomial=False, alphas=(0, 0.1, 1))
plt.subplot(224)
plot_model(Lasso, polynomial=True, alphas=(0, 10 ** -1, 1))
plt.show()

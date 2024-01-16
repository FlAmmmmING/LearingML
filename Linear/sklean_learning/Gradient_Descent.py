import matplotlib.pyplot as plt
import numpy as np

# 数据
X = 2 * np.random.rand(100, 1)
X_b = np.c_[np.ones((100, 1)), X]
y = 4 + 3 * X + np.random.rand(100, 1)
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]

# 学习率
eta = 0.1
# 迭代次数
n_iter = 1000
m = 100
# 批量梯度下降
theta = np.random.randn(2, 1)
for iteration in range(n_iter):
    gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
    theta -= eta * gradients
print(theta)

print(X_new_b.dot(theta))

# 随机梯度下降
theta_path_sgd = []
m = len(X_b)
n_epochs = 50
t0 = 5
t1 = 50


def learning_schedule(t):
    """
    Learning rate
    :param t: 迭代的 t 值
    :return:
    """
    return t0 / (t1 + t)


theta = np.random.randn(2, 1)
for epochs in range(n_epochs):
    for i in range(m):
        if epochs < 10 and i < 10:
            y_predict = X_new_b.dot(theta)
            plt.plot(X_new, y_predict, 'r--')
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index + 1]
        yi = y[random_index:random_index + 1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(n_epochs * m + i)
        theta = theta - eta * gradients
        theta_path_sgd.append(theta)
plt.plot(X, y, 'b.')
plt.axis([0, 2, 0, 15])
plt.show()


# 小批量梯度下降
theta_path_mgd = []
n_epochs = 50
minibatch_size = 16
theta = np.random.randn(2, 1)
t = 0
for epochs in range(n_epochs):
    shuffled_index = np.random.permutation(m)
    X_b_shuffled = X_b[shuffled_index]
    y_shuffled = y[shuffled_index]
    for i in range(0, m, minibatch_size):
        t += 1
        xi = X_b_shuffled[i:i+minibatch_size]
        yi = y_shuffled[i:i+minibatch_size]
        gradients = 2 / minibatch_size * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(t)
        theta = theta - eta * gradients
        theta_path_mgd.append(theta)
print(theta)

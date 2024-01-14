import numpy as np
from prepare_for_training import prepare_for_training


class LinearRegression:
    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
        """
        参数说明:
        :param data: training data
        :param labels:  supervised learning
        :param polynomial_degree: degrees of polynomial
        :param sinusoid_degree: degrees of sinusoidal
        :param normalize_data: whether to normalize
        """
        (
            data_processed,  # 预处理完后的数据
            features_mean,  # 得出的mean值
            features_deviation  # 得出的标准差
        ) = prepare_for_training(data, polynomial_degree, sinusoid_degree)

        self.data = data_processed
        self.labels = labels
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data

        num_features = self.data.shape[1]  # num_features 是这个样本有多少特征
        self.theta = np.zeros((num_features, 1))  # theta的个数和num_features个数一样

    def train(self, alpha, num_iterations=500):
        """
        学习率和迭代次数
        :param alpha: learning rate
        :param num_iterations: number of iterations
        :return: theta参数, cost history
        """
        cost_history = self.gradient_descent(alpha, num_iterations)
        return self.theta, cost_history

    def gradient_descent(self, alpha, num_iterations):
        """
        学习率和迭代次数
        :param alpha: learning rate
        :param num_iterations: number of iterations
        :return: cost history
        """
        cost_history = []  # 记录迭代损失值
        for _ in range(num_iterations):
            self.gradient_step(alpha)
            cost_history.append(self.cost_function(self.data, self.labels))
        return cost_history

    def cost_function(self, data, labels):
        """
        损失函数的计算
        :param data:
        :param labels:
        :return: cost
        """
        nums_example = data.shape[0]
        delta = LinearRegression.hypothesis(self.data, self.theta) - labels
        cost = 1 / 2 * np.dot(delta.T, delta) / nums_example
        return cost[0][0]

    def gradient_step(self, alpha):
        """
        梯度下降公式
        :param alpha: learning rate
        :return:
        """
        num_examples = self.data.shape[0]
        prediction = LinearRegression.hypothesis(self.data, self.theta)
        delta = prediction - self.labels  # 预测值-真实值
        theta = self.theta
        theta = theta - alpha * (1 / num_examples) * (np.dot(delta.T, self.data)).T
        self.theta = theta

    @staticmethod
    def hypothesis(data, theta):
        """
        :param data: training data
        :param theta: prediction theta
        :return: predictions
        """
        predictions = np.dot(data, theta)  # 公式: Prediction = t1 * d1 + t2 * d2 + t3 * d3 ...
        return predictions

    def get_cost(self, data, labels):
        data_processed = prepare_for_training(
            data,  # 预处理完后的数据
            self.polynomial_degree,
            self.sinusoid_degree,
            self.normalize_data
        )[0]

        return self.cost_function(data_processed, labels)

    def predict(self, data):
        """
        用训练的参数模型，与预测得到的回归值结果
        :param data:
        :return:
        """
        data_processed = prepare_for_training(
            data,  # 预处理完后的数据
            self.polynomial_degree,
            self.sinusoid_degree,
            self.normalize_data
        )[0]

        prediction = LinearRegression.hypothesis(data_processed, self.theta)

        return prediction

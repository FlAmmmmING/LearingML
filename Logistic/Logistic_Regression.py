import numpy as np
from scipy.optimize import minimize
from Linear.prepare_for_training import prepare_for_training


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class LogisticRegression:
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
        self.unique_labels = np.unique(self.labels)
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data

        num_features = self.data.shape[1]  # num_features 是这个样本有多少特征
        num_unique_labels = np.unique(labels).shape[0]
        self.theta = np.zeros((num_unique_labels, num_features))  # theta的个数和num_features个数一样

    def train(self, max_iterations=1000):
        cost_histories = []
        num_features = self.data.shape[1]
        for label_index, unique_label in enumerate(self.unique_labels):
            current_initial_theta = np.copy(self.theta[label_index].reshape(num_features, 1))
            current_labels = (self.labels == unique_label).astype(float)
            (current_labels, current_history) = LogisticRegression.gradient_descent(self.data, current_labels,
                                                                                    current_initial_theta,
                                                                                    max_iterations)

    @staticmethod
    def gradient_descent(data, labels, current_initial_theta, max_iterations):
        cost_histories = []
        num_features = data.shape[1]
        result = minimize(
            # 要优化的目标方程
            lambda current_theta: LogisticRegression.cost_function(data, labels,
                                                                   current_theta.reshape(num_features, 1)),
            # 初始化的权重参数
            current_initial_theta,
            # 选择优化策略
            method="CG",
            # 梯度下降迭代计算公式
            jac=lambda current_theta: LogisticRegression.gradient_step(data, labels, current_theta),
            # 记录结果
            callback=lambda current_theta: cost_histories.append(LogisticRegression.cost_function(data, labels, current_theta.reshape(num_features, 1))),
            # 迭代次数
            options={"maxiter": max_iterations}
        )
        if not result.success:
            raise ArithmeticError("Can not minimize the cost function" + result.message)
        optimized_theta = result.x.reshape(num_features, 1)
        return optimized_theta, cost_histories

    @staticmethod
    def cost_function(data, labels, theta):
        num_examples = data.shape[0]
        predictions = LogisticRegression.hypothesis(data, theta)
        # 总的样本损失
        y_is_set_cost = np.dot(labels[labels == 1].T, np.log(predictions[labels == 1]))
        y_is_not_set_cost = np.dot(1 - labels[labels == 0].T, np.log(1 - predictions[labels == 0]))
        cost = (-1 / num_examples) * (y_is_set_cost + y_is_not_set_cost)
        return cost

    @staticmethod
    def hypothesis(data, theta):
        predictions = sigmoid(np.dot(data, theta))
        return predictions

    @staticmethod
    def gradient_step(data, labels, current_theta):
        num_examples = labels.shape[0]
        predictions = LogisticRegression.hypothesis(data, current_theta)
        label_diff = predictions - labels
        gradient = (1 / num_examples) * np.dot(data.T, label_diff)
        return gradient.T.flatten()

    def predict(self, data):
        num_examples = data.shape[0]
        (
            data_processed,  # 预处理完后的数据
        ) = prepare_for_training(data, self.polynomial_degree, self.sinusoid_degree)[0]
        prob = LogisticRegression.hypothesis(data_processed, self.theta.T)
        max_prob_index = np.argmax(prob, axis=1)
        class_predictions = np.empty(max_prob_index.shape, dtype=object)
        for index, label in enumerate(self.unique_labels):
            class_predictions[max_prob_index == index] = label
        return class_predictions.reshape((num_examples, 1))

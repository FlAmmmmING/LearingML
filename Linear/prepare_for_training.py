import numpy as np
from sklearn.preprocessing import PolynomialFeatures


def normalize(features):
    features_normalized = np.copy(features).astype(float)
    features_mean = np.mean(features_normalized, 0)
    features_deviation = np.std(features, 0)
    if features.shape[0] > 1:
        features_normalized -= features_mean

    features_deviation[features_deviation == 0] = 1
    features_normalized /= features_deviation

    return features_normalized, features_mean, features_deviation


def generate_sinusoids(dataset, sinusoid_degree):
    num_examples = dataset.shape[0]
    sinusoids = np.empty((num_examples, 0))
    for degree in range(1, sinusoid_degree + 1):
        sinusoid_features = np.sin(degree * dataset)
        sinusoids = np.concatenate((sinusoids, sinusoid_features), axis=1)
    return sinusoids


"""数据预处理"""


def prepare_for_training(data, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
    # 计算样本总数
    num_examples = data.shape[0]

    data_processed = np.copy(data)

    # 预处理
    features_mean = 0
    features_deviation = 0
    data_normalized = data_processed
    if normalize_data:
        (
            data_normalized,
            features_mean,
            features_deviation
        ) = normalize(data_processed)

        data_processed = data_normalized

    # 特征变换sinusoid
    if sinusoid_degree > 0:
        sinusoids = generate_sinusoids(data_normalized, sinusoid_degree)
        data_processed = np.concatenate((data_processed, sinusoids), axis=1)

    # 特征变换polynomial
    if polynomial_degree > 0:
        polynomials = PolynomialFeatures(polynomial_degree)
        data_processed = np.concatenate((data_processed, polynomials), axis=1)

    # 加一列1
    data_processed = np.hstack((np.ones((num_examples, 1)), data_processed))

    return data_processed, features_mean, features_deviation

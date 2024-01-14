import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Linear_Regression import LinearRegression
import plotly
import plotly.graph_objs as go

plotly.offline.init_notebook_mode()
data = pd.read_csv('./dataset/2019.csv')
# 分割训练集和测试集
train_data = data.sample(frac=0.8)
test_data = data.drop(train_data.index)

input_param_name1 = 'GDP per capita'
input_param_name2 = 'Freedom to make life choices'
output_param_name = 'Healthy life expectancy'

X_train = train_data[[input_param_name1, input_param_name2]].values
Y_train = train_data[[output_param_name]].values

X_test = test_data[[input_param_name1, input_param_name2]].values
Y_test = test_data[[output_param_name]].values

plot_training_trace = go.Scatter3d(
    x=X_train[:, 0].flatten(),
    y=X_train[:, 1].flatten(),
    z=Y_train.flatten(),
    name="Training Set",
    mode="markers",
    marker={
        'size': 10,
        'opacity': 1,
        'line': {
            'color': 'rgb(255, 255, 255)',
            'width': 1
        }
    }
)

plot_test_trace = go.Scatter3d(
    x=X_test[:, 0].flatten(),
    y=X_test[:, 1].flatten(),
    z=Y_test.flatten(),
    name="Test Set",
    mode="markers",
    marker={
        'size': 10,
        'opacity': 1,
        'line': {
            'color': 'rgb(255, 255, 255)',
            'width': 1
        }
    }
)

plot_layout = go.Layout(
    title='Data Set',
    scene={
        'xaxis': {'title': input_param_name1},
        'yaxis': {'title': input_param_name2},
        'zaxis': {'title': output_param_name},
    },
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
)

plot_data = [plot_training_trace, plot_test_trace]
plot_figure = go.Figure(data=plot_data, layout=plot_layout)
plotly.offline.plot(plot_figure)

num_iterations = 500
learning_rate = 0.01
polynomial_degree = 0
sinusoidal_degree = 2

model = LinearRegression(X_train, Y_train, polynomial_degree, sinusoidal_degree)
theta, cost_history = model.train(
    learning_rate, num_iterations
)

print('开始时的损失', cost_history[0])
print('训练后的损失', cost_history[-1])

plt.plot(range(num_iterations), cost_history)
plt.xlabel('iterations')
plt.ylabel('Cost')
plt.title('Gradient Descent Progress')
plt.show()

# 测试
predictions_nums = 100
X_min = X_train[:, 0].min()
X_max = X_train[:, 0].max()
Y_min = X_train[:, 1].min()
Y_max = X_train[:, 1].max()

X_axis = np.linspace(X_min, X_max, predictions_nums)
Y_axis = np.linspace(Y_min, Y_max, predictions_nums)

X_predictions = np.zeros((predictions_nums * predictions_nums, 1))
Y_predictions = np.zeros((predictions_nums * predictions_nums, 1))
X_Y_index = 0
for X_index, X_value in enumerate(X_axis):
    for Y_index, Y_value in enumerate(Y_axis):
        X_predictions[X_Y_index] = X_value
        Y_predictions[X_Y_index] = Y_value
        X_Y_index += 1
Z_predictions = model.predict(np.hstack((X_predictions, Y_predictions)))
plot_predictions_trace = go.Scatter3d(
    x=X_predictions.flatten(),
    y=Y_predictions.flatten(),
    z=Z_predictions.flatten(),
    name='Predictions Planes',
    mode='markers',
    marker={
        'size': 1,
    },
    opacity=0.8,
    surfaceaxis=2,
)

plot_data = [plot_training_trace, plot_test_trace, plot_predictions_trace]
plot_figure = go.Figure(data=plot_data, layout=plot_layout)
plotly.offline.plot(plot_figure)

# X_predictions = np.linspace(X_train.min(), X_train.max(), predictions_nums).reshape(predictions_nums, 1)
# Y_predictions = model.predict(X_predictions)
#
# plt.scatter(X_train, Y_train, label='Train_data')
# plt.scatter(X_test, Y_test, label='Test_data')
# plt.plot(X_predictions, Y_predictions, 'r', label='prediction')
# plt.xlabel(input_param_name1)
# plt.ylabel(output_param_name)
# plt.title("Data")
# plt.legend()
# plt.show()

## https://www.evergreeninnovations.co/blog-quantile-loss-function-for-machine-learning/
## https://towardsdatascience.com/quantile-regression-from-linear-models-to-trees-to-deep-learning-af3738b527c3
## https://shrmtmt.medium.com/quantile-loss-in-neural-networks-6ea215fcee99
## https://scikit-learn.org/stable/modules/neural_networks_supervised.html

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def f_predictable(x):
    return x+np.sin(np.pi*x/2)


def f(x, std=0.2):
    return f_predictable(x)+np.random.randn(len(x))*std


def get_data(num, start=0, end=4):
        x = np.sort(np.random.rand(num)*(end-start)+start)
        y = f(x)
        return x.reshape(-1, 1), y

x_train, y_train = get_data(num=20000)
x_test, y_test = get_data(num=1000)

def quantile_loss(q, y, y_p):
        e = y-y_p
        return tf.keras.backend.mean(tf.keras.backend.maximum(q*e, (q-1)*e))

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(100, activation='relu', input_dim=1))
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='linear'))
# The lambda function is used to input the quantile value to the quantile
# regression loss function. Keras only allows two inputs in user-defined loss
# functions, actual and predicted values.
quantile = 0.977
model.compile(optimizer='adam', loss=lambda y, y_p: quantile_loss(quantile, y, y_p))
model.fit(x_train, y_train, epochs=1)
# ~ model.fit(x_train, y_train, epochs=20)
prediction = model.predict(x_test)

myline = np.linspace(0, 4.0, 100)

plt.scatter(x_train, y_train)
plt.plot(myline, model(myline))
plt.show()

# ~ # Set the model to evaluation mode
# ~ model.eval()

# ~ # Predict the quantiles
# ~ with torch.no_grad():
    # ~ predictions = model(x_train)

# ~ # Convert the predictions and x_train to numpy for plotting
# ~ x_train_np = x_tensor.numpy().flatten()
# ~ y_train_np = y_train.numpy().flatten()
# ~ predictions_np = predictions.numpy()

# ~ # Plotting
# ~ plt.figure(figsize=(12, 6))
# ~ plt.scatter(x_train_np, y_train_np, label='Actual Data', color='blue', marker='.', alpha=0.1)
# ~ plt.scatter(x_train_np, predictions_np[:, 0], label=f'{QUANTILES[0]:.4f} Percentile', color='green', marker='.', alpha=0.1)
# ~ plt.scatter(x_train_np, predictions_np[:, 1], label=f'{QUANTILES[1]:.4f} Percentile', color='red', marker='.', alpha=0.1)
# ~ plt.scatter(x_train_np, predictions_np[:, 2], label=f'{QUANTILES[2]:.4f} Percentile', color='purple', marker='.', alpha=0.1)
# ~ plt.scatter(x_train_np, predictions_np[:, 3], label=f'{QUANTILES[3]:.4f} Percentile', color='orange', marker='.', alpha=0.1)
# ~ plt.title('Predicted Quantiles vs. Actual Data')
# ~ plt.xlabel('x_train')
# ~ plt.ylabel('y_train and Predicted Quantiles')
# ~ plt.legend()
# ~ plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Hyperparameters
epochs = 1000
learning_rate = 1e-2
# Simulating random linear corellation
x = np.linspace(1, 20)
m = np.random.uniform(0, 4)
c = np.random.uniform(0, 10)
y = x * m + c
epsilon = np.random.randn(50)
y = np.add(y, epsilon)

plt.figure()
plt.scatter(x, y)
plt.xlabel("Input space")
plt.ylabel("Output space")
plt.grid()
plt.show()

class LinearRegressor():
    def __init__(self):
        self.beta = np.random.randn(2) # beta[0] is intercept, beta[1] is slope

    def forward(self, X):
        return X * self.beta[1] + self.beta[0]

    def loss(self, Y, y_hat):
        return (y_hat - Y)**2

    def grad_descent(self, X, Y, lr = 1e-3):  # All functions only for singular input/output
        y_hat = self.forward(X)
        self.beta[0] -= 2 * (y_hat - Y) * lr
        self.beta[1] -= 2 * (y_hat - Y) * X * lr
        return

model = LinearRegressor()


def plot(epoch):
    plt.figure()
    plt.scatter(x, y, label = "data")
    y_hat = x * model.beta[1] + model.beta[0]
    plt.plot(x, y_hat, label = "Regression line")
    plt.legend()
    plt.xlabel("Input space")
    plt.ylabel("Output space")
    plt.grid()
    plt.show()

for e in range(epochs):
    if (e + 1) % 100 == 0:
        plot(e+1)
    for sample in range(len(x)):
        model.grad_descent(x[sample], y[sample], lr = learning_rate)

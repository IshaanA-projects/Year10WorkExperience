import numpy as np
import matplotlib.pyplot as plt
class PolynomialRegressor:

    def __init__(self, degree):
        self.m = degree

    def Regression(self, X, Y):
        a = np.zeros((self.m + 1, self.m + 1))
        b = np.zeros((self.m + 1, 1))

        for p in range(self.m + 1):
            for q in range(self.m + 1):
                a[p][q] = np.sum([X[i] ** (p + q) for i in range(len(X))])
            b[p][0] = np.sum([X[i] ** p * Y[i] for i in range(len(X))])

        self.beta = np.linalg.solve(a, b)

    def prediction(self, X):
        design_X = np.zeros((len(X), self.m + 1))
        for p in range(len(X)):
            for q in range(self.m + 1):
                design_X[p][q] = X[p] ** q
        return (design_X @ self.beta).reshape(len(X))

    def RMSE(self, X, Y): # Root mean squared error (Root brings back into same unit as target)
        return np.sqrt(1 / len(X) * np.sum((model.prediction(X) - Y) ** 2))


def polynomial(X, coeffs):
    design_X = np.zeros((len(X), len(coeffs)))
    for p in range(len(X)):
        for q in range(len(coeffs)):
            design_X[p][q] = X[p] ** q
    return design_X @ coeffs

n = 50
x = np.linspace(-20, 20, 50)
deg = np.random.randint(1, 10)
coeficients = np.random.randn(deg + 1)
y = polynomial(x, coeficients)

plt.plot(x, y)
plt.show()

losses = np.array([])
for degree in range(1, 10):
    model = PolynomialRegressor(degree)
    model.Regression(x, y)

    losses = np.append(losses, model.RMSE(x, y))

degree = np.argmin(losses) + 1

model = PolynomialRegressor(degree)
model.Regression(x, y)

plt.scatter(x, model.prediction(x), c = "red")
plt.scatter(x, y, c = "green")
plt.show()

print(f"The optimal polynomial degree is {degree}")
print(f"The actual degree is {deg}")

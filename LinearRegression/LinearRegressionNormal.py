import numpy as np
from sklearn.linear_model import LinearRegression

X = np.linspace(1, 50, 100)
m = np.random.randn()
c = np.random.randn()
epsilon = np.random.randn(100)
y = X * m + c
y += epsilon

# With numpy

design_X = np.ones((len(X), 2))
for i in range(len(X)):
    row = design_X[i]
    row[0] = X[i]

np_beta = np.linalg.inv(design_X.T @ design_X) @ design_X.T @ y


# With sklearn


X = X.reshape(-1, 1)
y = y.reshape(-1, 1)
reg = LinearRegression().fit(X, y)

sklearn_beta = np.append(reg.coef_[0], reg.predict(np.array([0]).reshape(-1, 1)))

print(f"Coefficients from numpy : {np_beta}")
print(f"Coefficients from sklearn : {sklearn_beta}")

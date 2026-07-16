import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import scale
import torch.optim as optim

window_length = 50
prediction_gap = 20
split = 0.8
epochs = 2000

# Importing data

df = pd.read_csv(r"BA.csv")

# Removing unecessary data and adding useful statsitics

df["Volatility"] = (df["High"] - df["Low"]) / df["Close"]
df["PCT_change"] = (df["Adj Close"] - df["Adj Close"].shift(100)) / df["Adj Close"] # Percentage change on long timescale, gives good idea of trend


df.dropna(inplace = True)
df = df[["Adj Close", "Volatility", "Volume", "PCT_change"]]

X = scale(df.drop(["Adj Close"], axis = 1))
y = df["Adj Close"]

class StockPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = nn.Linear(3 * window_length, 200)
        self.W2 = nn.Linear(200, 200)
        self.W3 = nn.Linear(200, 1)

    def forward(self, x):
        x = F.leaky_relu(self.W1(x))
        x = F.leaky_relu(self.W2(x))
        x = self.W3(x)
        return x

def create_sequences(list, w = window_length):
    x = []
    for i in range(w, len(list)):
        x.append(list[i - w : i])
    return x

def prepro(x):
    x = np.reshape(x, (-1, 3 * window_length))
    return x



X = np.array(create_sequences(X))
X = prepro(X)
y = y[window_length:]  # To keep y and X aligned and of the same length
y = np.array(y)

idx = int(len(X) * split)
X_train = X[:idx]
X_test = X[idx:]
y_train = y[:idx]
y_test = y[idx:]

model = StockPredictor()
optimizer = optim.Adam(model.parameters())
criterion = nn.MSELoss()

losses = []
e_loss = 0
for i in range(epochs):
    optimizer.zero_grad()
    if (i + 1) % 100 == 0:
        if i > 0:
            losses.append(e_loss)
            print(e_loss)
            print(f"Epoch : {i + 1}")
            print(f"Test loss = {criterion(model.forward(torch.tensor(X_test, dtype=torch.float32)), torch.tensor(y_test, dtype=torch.float32))}")
        e_loss = 0

    x_sample = torch.tensor(X_train, dtype=torch.float32)
    y_sample = torch.tensor(y_train, dtype=torch.float32)
    y_sample = torch.reshape(y_sample, (-1, 1))
    y_hat = model.forward(x_sample)
    loss = criterion(y_hat, y_sample)
    e_loss += loss
    loss.backward()
    optimizer.step()

print(losses)

plt.figure()
plt.plot(y_test, label = "Actual")
plt.legend()
plt.plot(model.forward(torch.tensor(X_test, dtype=torch.float32)).detach().numpy(), label = "Predicted")
plt.legend()
plt.show()

print(f"Test loss = {criterion(model.forward(torch.tensor(X_test, dtype=torch.float32)), torch.tensor(y_test, dtype=torch.float32))}")


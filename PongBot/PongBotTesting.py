import numpy as np
import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Bernoulli
import torch.nn.functional as F

episodes = 5

class PongBot(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = nn.Linear(256, 200)
        self.W2 = nn.Linear(200, 100)
        self.W3 = nn.Linear(100, 1)


    def forward(self, x):
        x = F.relu(self.W1(x))
        x = F.relu(self.W2(x))
        x = F.sigmoid(self.W3(x))
        return x


def prepro(I):
    I = torch.from_numpy(I)
    I = I.to(torch.float)
    I = I / 255.0
    I = I.reshape(1, -1)
    return I



model = PongBot()
model = torch.load(r"bot.pt", weights_only = False)
model.eval()


gym.register_envs(ale_py)
env = gym.make("ALE/Pong-v5", render_mode = "human", obs_type = "ram")


for i in range(episodes):
    state = prepro(env.reset()[0])
    prev_state = torch.zeros_like(state)
    
    done = False
    
    while not done:
        state_input = torch.cat((state, prev_state), dim=1)
        with torch.no_grad():
            prob = model(state_input)
            action = Bernoulli(prob).sample()
        action = action.item() + 2
        prev_state = state.clone()
        step = env.step(action)
        state = prepro(step[0])
        reward = step[1]
        terminated = step[2]
        truncated = step[3]
        done = terminated or truncated

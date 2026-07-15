import numpy as np
import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Bernoulli
import torch.nn.functional as F

episode_length = 1000
class PongBot(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = nn.Linear(128, 200)
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
    I = I.reshape(1, -1)
    return I



model = PongBot()
model = torch.load(r"bot.pt", weights_only = False)
model.eval()


gym.register_envs(ale_py)
env = gym.make("ALE/Pong-v5", render_mode = "human", obs_type = "ram")

state = prepro(env.reset()[0])
init_state = prepro(np.zeros(128))
state_pool = []
action_pool = []
reward_pool = []

for j in range(episode_length):
    if j == 0:
        prob = model.forward(init_state)
        m = Bernoulli(prob).sample()
        action = m.numpy().astype(int).item() + 2
        state_pool.append(state)
        step = env.step(action)
        state = prepro(step[0])
        reward = step[1]
        action_pool.append(action)
        reward_pool.append(reward)
    else:
        prob = model.forward(state - state_pool[j - 1])
        m = Bernoulli(prob).sample()
        action = m.numpy().astype(int).item() + 2
        state_pool.append(state)
        step = env.step(action)
        state = prepro(step[0])
        reward = step[1]
        action_pool.append(action)
        reward_pool.append(reward)

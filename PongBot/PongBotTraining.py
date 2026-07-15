import numpy as np
import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Bernoulli
import torch.nn.functional as F



discount_factor = 0.99
alpha = 1e-3
episodes = 10000
episode_length = 200
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
    I.reshape(1, -1)
    return I


gym.register_envs(ale_py)
env = gym.make("ALE/Pong-v5", obs_type = "ram")

model = PongBot()

optimizer = optim.SGD(model.parameters(), lr = alpha, momentum = 0.9)

for i in range(episodes):
    # New episode
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
            prob = model.forward(state - state_pool[-1])
            m = Bernoulli(prob).sample()
            action = m.numpy().astype(int).item() + 2
            state_pool.append(state)
            step = env.step(action)
            state = prepro(step[0])
            reward = step[1]
            action_pool.append(action)
            reward_pool.append(reward)

    # Discounted reward system
    running_reward = 0
    for r in reversed(range(episode_length)):
        if reward_pool[r] == -1:
            running_reward = 0
        running_reward = running_reward * discount_factor + reward_pool[r]
        reward_pool[r] = running_reward



    # Normalising the reward pool (So that half of actions are punished and half are rewarded)
    reward_pool -= np.mean(reward_pool)
    reward_pool /= np.max([np.std(reward_pool), 1e-2])

    # Gradient descent
    losses = 0
    for s in range(episode_length):
        if s != 0:
            optimizer.zero_grad()
            state = state_pool[s] - state_pool[s - 1]
            prob = model.forward(state)
            m = Bernoulli(prob)
            action  = np.array(([action_pool[s] - 2])).astype(float)
            action = torch.tensor(action)
            reward = reward_pool[s]

            loss = -1 * m.log_prob(action) * reward
            losses += loss.item()
            loss.backward()

        else:
            optimizer.zero_grad()
            state = prepro(np.zeros(128))
            prob = model.forward(state)
            m = Bernoulli(prob)
            action = np.array(([action_pool[s] - 2])).astype(float)
            action = torch.tensor(action)
            reward = reward_pool[s]

            loss = -1 * m.log_prob(action) * reward
            losses += loss.item()
            loss.backward()




    optimizer.step()

    if (i + 1) % 500 == 0:
        print(f"Episode : {i+1}")
        print(losses)


torch.save(model, r"bot.pt")

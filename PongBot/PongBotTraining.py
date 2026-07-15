import numpy as np
import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Bernoulli
import torch.nn.functional as F



discount_factor = 0.99
alpha = 1e-4
episodes = 10000
batch_size = 10


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


gym.register_envs(ale_py)
env = gym.make("ALE/Pong-v5", obs_type = "ram")

model = PongBot()

optimizer = optim.RMSprop(model.parameters(), lr = alpha)

for i in range(episodes):
    # New episode
    if i % batch_size == 0:
        optimizer.zero_grad()
        losses = 0
    state = prepro(env.reset()[0])
    init_state = prepro(np.zeros(128))
    state_pool = []
    action_pool = []
    reward_pool = []
    done = False
    first_step = True
    while not done:
        if first_step:
            state_difference = init_state
            first_step = False
        else:
            state_difference = state - state_pool[-1]
        prob = model.forward(state_difference)
        m = Bernoulli(prob).sample()
        action = m.numpy().astype(int).item() + 2
        state_pool.append(state)
        step = env.step(action)
        state = prepro(step[0])
        reward = step[1]
        terminated = step[2]
        truncated = step[3]
        done = terminated or truncated
        action_pool.append(action)
        reward_pool.append(reward)

    # Discounted reward system
    running_reward = 0
    for r in reversed(range(len(reward_pool))):
        if reward_pool[r] != 0:
            running_reward = 0
        running_reward = running_reward * discount_factor + reward_pool[r]
        reward_pool[r] = running_reward



    # Normalising the reward pool (So that half of actions are punished and half are rewarded)
    reward_pool = np.array(reward_pool)
    reward_pool -= np.mean(reward_pool)
    reward_pool /= np.max([np.std(reward_pool), 1e-2])

    # Gradient descent
    for s in range(len(reward_pool)):
        if s == 0:
            state_difference = init_state
        else:
            state_difference = state_pool[s] - state_pool[s-1]

        prob = model.forward(state_difference)
        m = Bernoulli(prob)
        action = torch.tensor([[action_pool[s] - 2]])
        reward = reward_pool[s]

        loss = -1 * m.log_prob(action) * reward
        losses += loss.item()
        loss.backward()

    if (i + 1) % batch_size == 0:
        optimizer.step()
        print(losses)

    if (i + 1) % 500 == 0:
        print(f"Episode : {i+1}")


torch.save(model, r"bot.pt")

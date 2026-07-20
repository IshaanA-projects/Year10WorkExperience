import numpy as np
import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Bernoulli

discount_factor = 0.99
alpha = 1e-4
episodes = 20000
batch_size = 10
raw_reward = []

class PongBot(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = nn.Linear(256, 200)
        self.W2 = nn.Linear(200, 100)
        self.W3 = nn.Linear(100, 1)


    def forward(self, x):
        x = torch.relu(self.W1(x))
        x = torch.relu(self.W2(x))
        x = torch.sigmoid(self.W3(x))
        return x

class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = nn.Linear(256, 200)
        self.W2 = nn.Linear(200, 200)
        self.W3 = nn.Linear(200, 1)

    def forward(self, x):
        x = torch.relu(self.W1(x))
        x = torch.relu(self.W2(x))
        x = self.W3(x)
        return x

def prepro(I):
    I = torch.from_numpy(I).float()
    I = I / 255.0
    I = I.reshape(1, -1)
    return I


gym.register_envs(ale_py)
env = gym.make("ALE/Pong-v5", obs_type = "ram")

actor = torch.load(r"actor.pt", weights_only=False)
critic = torch.load(r"critic.pt", weights_only=False)
actor_optimizer = optim.RMSprop(actor.parameters(), lr = alpha)
critic_optimizer = optim.RMSprop(critic.parameters(), lr = alpha)
mse = nn.MSELoss()

for i in range(episodes):
    # New episode
    if i % batch_size == 0:
        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()
        critic_losses = []

    state = prepro(env.reset()[0])
    prev_state = torch.zeros_like(state)
    state_pool = []
    action_pool = []
    reward_pool = []
    done = False

    while not done:
        state_input = torch.cat((state, prev_state), dim = 1)
        with torch.no_grad():
            prob = actor(state_input)
            m = Bernoulli(prob).sample()
        action = int(m.item()) + 2
        state_pool.append(state.clone().detach())
        prev_state = state.clone().detach()
        step = env.step(action)
        state = prepro(step[0])
        reward = step[1]
        terminated = step[2]
        truncated = step[3]
        done = terminated or truncated
        action_pool.append(action)
        reward_pool.append(reward)

    raw_reward.append(sum(reward_pool))

    # Discounted reward system
    running_reward = 0
    for r in reversed(range(len(reward_pool))):
        if reward_pool[r] != 0:
            running_reward = 0
        running_reward = running_reward * discount_factor + reward_pool[r]
        reward_pool[r] = running_reward

    # Gradient descent
    for s in range(len(reward_pool)):
        if s == 0:
            prev_state = torch.zeros_like(state_pool[s])
        else:
            prev_state = state_pool[s-1]

        state_input = torch.cat((state_pool[s], prev_state), dim = 1)
        prob = actor(state_input)
        m = Bernoulli(prob)
        action = torch.tensor([[float(action_pool[s] - 2)]])
        reward = torch.tensor([[float(reward_pool[s])]])
        baseline = critic(state_input)
        critic_loss = mse(baseline, reward)
        advantage = reward - baseline.detach()
        actor_loss = -1 * m.log_prob(action) * advantage
        critic_losses.append(critic_loss.item())
        loss = actor_loss + critic_loss
        loss.backward()

    if (i + 1) % batch_size == 0:
        print(f"Episode : {i + 1}")
        actor_optimizer.step()
        critic_optimizer.step()
        print(f"Critic loss : {np.mean(critic_losses)}")

    if (i + 1) % 100 == 0:
        print(f"Average Reward : {sum(raw_reward) / len(raw_reward)}")
        print(f"Highest Reward : {max(raw_reward)}")
        print(f"Lowest Reward : {min(raw_reward)}")
        raw_reward = []
    if (i + 1) % 500 == 0:   # Regular model saving to allow for the program to be run without fear of losing model parameters
        torch.save(actor, r"actor.pt")
        torch.save(critic, r"critic.pt")


torch.save(actor, r"actor.pt")
torch.save(critic, r"critic.pt")

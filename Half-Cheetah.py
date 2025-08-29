import torch
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
import gymnasium as gym

env = gym.make("HalfCheetah-v5")
OBS_SPACE = env.observation_space.shape[0]
ACTION_SPACE = env.action_space.shape[0]

device = "mps" if torch.mps.is_available() else "cpu"
device = "cuda" if torch.cuda.is_available() else device
device = "cpu"

class OUP():
    def __init__(self, theta = 0.15, sigma=0.2, dim = ACTION_SPACE):
        self.dim = dim
        self.theta = -theta
        self.sigma = sigma
        self.x_t = np.zeros(self.dim, dtype = np.float32)

    def reset(self):
        self.x_t = np.zeros(self.dim, dtype = np.float32)

    def __call__(self):
        dW_t = np.random.randn(self.dim)
        dx_t = self.theta * self.x_t + self.sigma * dW_t
        self.x_t += dx_t
        return self.x_t
    
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(OBS_SPACE, 128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,ACTION_SPACE),
            nn.Tanh()
        )
    
    def forward(self, state):
        return self.model(state)
    
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(OBS_SPACE + ACTION_SPACE, 128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )

    def forward(self, state, action):
        state_action = torch.cat((state,action), dim = -1)
        return self.model(state_action)
    
oup = OUP()

actor = Actor().to(device)
actor_optim = optim.Adam(actor.parameters(), 1e-4)
actor_target = Actor().to(device)
actor_target.load_state_dict(actor.state_dict())

critic = Critic().to(device)
critic_optim = optim.Adam(critic.parameters(), 1e-3)
critic_target = Critic().to(device)
critic_target.load_state_dict(critic.state_dict())

mse = nn.MSELoss()

GAMMA = 0.99
TAU = 0.1
NUM_EPOCHS = 1000
BATCH_SIZE = 500
EXPERIENCE_REPLAY_SIZE = 300000

experience_replay = deque([], maxlen = EXPERIENCE_REPLAY_SIZE)
reward_over_time = []

for epoch in range(NUM_EPOCHS):
    state, _ = env.reset()
    state = torch.tensor(state, dtype = torch.float32).to(device)

    oup.reset()
    reward_during_epoch = 0

    while True:
        with torch.no_grad():
            action = actor(state).cpu() + oup()
            next_state, reward, terminated, truncated, _ = env.step(np.array(action.cpu()))
            next_state = torch.tensor(next_state, dtype = torch.float32).to(device)
            experience_replay.append((state,action.to(device), reward, next_state))
            reward_during_epoch += reward
            state = next_state

        if len(experience_replay) > 10000:
            with torch.no_grad():
                random_sample = random.sample(experience_replay, BATCH_SIZE)
                state_array, action_array, reward_array, next_state_array = zip(*random_sample)
                
                state_array = torch.stack(state_array)
                action_array = torch.stack(action_array)
                reward_array = torch.tensor(reward_array, dtype = torch.float32).to(device)
                next_state_array = torch.stack(next_state_array)

                #update critc
                target_actions = actor_target(next_state_array).to(device)
                target_state_values = reward_array.unsqueeze(1) + GAMMA * critic_target(next_state_array, target_actions).to(device)
            predicted_state_values = critic(state_array, action_array)

            critic_optim.zero_grad()
            critic_loss = mse(predicted_state_values, target_state_values)
            critic_loss.backward()
            critic_optim.step()

            #update actor
            actor_optim.zero_grad()
            new_actions = actor(state_array).to(device)
            actor_loss = -1 * critic(state_array, new_actions).mean()
            actor_loss.backward()
            actor_optim.step()

            #update target actor
            new_state_dict_for_actor_target = {}
            for key in actor.state_dict().keys():
                new_state_dict_for_actor_target[key] = TAU * actor.state_dict()[key] + (1 - TAU) * actor_target.state_dict()[key]
            actor_target.load_state_dict(new_state_dict_for_actor_target)

            #update target critic
            new_state_dict_for_critic_target = {}
            for key in critic.state_dict().keys():
                new_state_dict_for_critic_target[key] = TAU * critic.state_dict()[key] + (1 - TAU) * critic_target.state_dict()[key]
            critic_target.load_state_dict(new_state_dict_for_critic_target)
        
        if terminated or truncated:
            break

    reward_over_time.append(reward_during_epoch)

    if epoch % 10 == 0 and epoch != 0:
        print(f"Epoch {epoch}: Current Reward {reward_over_time[-1]:.2f}, Average Reward (Last 50) {np.mean(reward_over_time[max(0, epoch-50):]):.2f}")



plt.plot(reward_over_time, color = "blue", label = "Rewards During Training")
plt.plot([np.mean(reward_over_time[max(0,epoch - 50):epoch]) for epoch in range(NUM_EPOCHS)], color = "red", label = "Average Reward")
plt.grid()
plt.legend()
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Half Cheetah Training")
plt.show()

            






        



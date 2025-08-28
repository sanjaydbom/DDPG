import torch
import gymnasium as gym
from torch import nn, optim
import matplotlib.pyplot as plt
import numpy as np 
import random
from collections import deque

env = gym.make("Pendulum-v1")

class OUP():
    def __init__(self, theta = 0.15, sigma = 0.2):
        self.x_t = 0
        self.theta = -1 * theta
        self.sigma = sigma

    def __call__(self):
        dW_t = np.random.randn()
        dx_t = self.theta * self.x_t + self.sigma * dW_t
        self.x_t += dx_t
        return self.x_t
    
    def reset(self):
        self.x_t = 0

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(3,32)
        self.actor = nn.Linear(32,1)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.layer1(x)
        x = self.relu(x)
        return 2.0 * torch.tanh(self.actor(x))
    
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(4,32)
        self.actor = nn.Linear(32,1)
        self.relu = nn.ReLU()

    def forward(self,state, action):
        x = torch.cat((state, action), -1)
        x = self.layer1(x)
        x = self.relu(x)
        return self.actor(x)
    
actor = Actor()
actor_optimizer = optim.Adam(actor.parameters(), 1e-4)
actor_target = Actor()
actor_target.load_state_dict(actor.state_dict())

critic = Critic()
critic_optimizer = optim.Adam(critic.parameters(), 1e-3)
critic_target = Critic()
critic_target.load_state_dict(critic.state_dict())

GAMMA = 0.99
TAU = 0.001
NUM_EPOCHS = 1000
BATCH_SIZE = 50
EXPERIENCE_REPLAY_LENGTH = 30000

mse_loss = nn.MSELoss()

experience_replay = deque([], maxlen=EXPERIENCE_REPLAY_LENGTH)

oup = OUP()

rewards_during_training = [0] * 1000
for epoch in range(NUM_EPOCHS):
    state, _ = env.reset()
    oup.reset()
    reward_during_single_episode = 0

    while True:
        with torch.no_grad():
            state = torch.tensor(state, dtype = torch.float32)
            action = actor(state) + oup()
            next_state, reward, terminated, truncated, _ = env.step(np.array(action))
            experience_replay.append((state, action, reward, next_state))
            reward_during_single_episode += reward

        if len(experience_replay) > 100:
            random_replays = random.sample(experience_replay, BATCH_SIZE)
            state_array_from_replays, action_array_from_replays, reward_arrays_from_replays, next_state_from_replays = zip(*random_replays)
            state_array_from_replays = torch.stack(state_array_from_replays)
            action_array_from_replays = torch.stack(action_array_from_replays)
            reward_arrays_from_replays = torch.tensor(reward_arrays_from_replays, dtype = torch.float32)
            next_state_array_from_replays = torch.stack([torch.tensor(next_individual_state_from_replays) for next_individual_state_from_replays in next_state_from_replays])
            
            target_actions = actor_target(next_state_array_from_replays).detach()
            actual_state_values = reward_arrays_from_replays.unsqueeze(1) + GAMMA * critic_target(next_state_array_from_replays, target_actions)
            predicted_state_values = critic(state_array_from_replays, action_array_from_replays)


            critic_optimizer.zero_grad()
            critic_loss = mse_loss(actual_state_values.detach(), predicted_state_values)
            critic_loss.backward()
            critic_optimizer.step()

            actor_actions = actor(state_array_from_replays)
            actor_loss = -critic(state_array_from_replays, actor_actions).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            new_actor_target_weights = {}
            for key in actor.state_dict().keys():
                new_actor_target_weights[key] = TAU * actor.state_dict()[key] + (1 - TAU) * actor_target.state_dict()[key]
            actor_target.load_state_dict(new_actor_target_weights)

            new_critic_target_weights = {}
            for key in critic.state_dict().keys():
                new_critic_target_weights[key] = TAU * critic.state_dict()[key] + (1 - TAU) * critic_target.state_dict()[key]
            critic_target.load_state_dict(new_critic_target_weights)

        
        state = next_state
        if terminated or truncated:
            rewards_during_training[epoch] = reward_during_single_episode
            break
    
    if epoch % 10 == 0 and epoch != 0:
        print(f"Epoch {epoch}: Last Reward {rewards_during_training[epoch]:.2f}, Avg Reward (Last 50) {np.mean(rewards_during_training[max(0,epoch - 50):epoch+1]):.2f}")

plt.plot(rewards_during_training, color = "blue", label = "Rewards During Training")
plt.plot([np.mean(rewards_during_training[max(0,epoch - 50):epoch]) for epoch in range(NUM_EPOCHS)], color = "red", label = "Average Reward")
plt.grid()
plt.legend()
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()
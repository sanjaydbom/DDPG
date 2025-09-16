import torch
from torch import nn

from OlensteinUlenbeckProcess import OUP

class Actor(nn.Module):
    def __init__(self, obs_space, action_space, hidden_layers):
        super(Actor, self).__init__()
        self.head = nn.Linear(obs_space, hidden_layers[0])
        self.hidden = nn.ModuleList([nn.Linear(hidden_layers[i], hidden_layers[i+1]) for i in range(len(hidden_layers)-1)])
        self.tail = nn.Linear(hidden_layers[-1], action_space)
        self.relu = nn.ReLU()

    def forward(self, state):
        x = self.head(state)
        x = self.relu(x)
        for layer in self.hidden:
            x = layer(x)
            x = self.relu(x)
        x = self.tail(x)

        return x
    
class Critic(nn.Module):
    def __init__(self, obs_space, action_space, hidden_layers):
        super(Critic, self).__init__()
        self.head = nn.Linear(obs_space+action_space, hidden_layers[0])
        self.hidden = nn.ModuleList([nn.Linear(hidden_layers[i], hidden_layers[i+1]) for i in range(len(hidden_layers)-1)])
        self.tail = nn.Linear(hidden_layers[-1], 1)
        self.relu = nn.ReLU()

    def forward(self, state, action):
        x = torch.cat((state, action), dim = -1)
        x = self.head(x)
        x = self.relu(x)
        for layer in self.hidden:
            x = layer(x)
            x = self.relu(x)
        x = self.tail(x)

        return x
    
def get_architecture(obs_space, action_space, actor_hidden, critic_hidden):
    actor = Actor(obs_space,actor_hidden,actor_hidden)
    target_actor = Actor(obs_space,actor_hidden,actor_hidden)
    target_actor.load_state_dict(actor.state_dict())
    critic = Critic(obs_space,action_space,critic_hidden)
    target_critic = Critic(obs_space,action_space,critic_hidden)
    target_critic.load_state_dict(critic.state_dict())
    return actor, target_actor, critic, target_critic
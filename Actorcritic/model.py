import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from collections import namedtuple, deque

BUFFER_SIZE = int(1e6)  
BATCH_SIZE = 1024        
LR = 0.02
betas = (0.9, 0.999)
GAMMA = 0.99
UPDATE_EVERY = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ActorCritic(nn.Module):
    def __init__(self, input_shape, action_size, fc1, fc2):
        super(ActorCritic, self).__init__()
        # Actor network
        self.fc1 = nn.Sequential(nn.Linear(input_shape, fc1), nn.ReLU())
        self.action_layer = nn.Sequential(nn.Linear(fc1, action_size), nn.Softmax())

        #Critic network
        self.fc2 = nn.Sequential(nn.Linear(input_shape,fc2), nn.ReLU())
        self.value_layer = nn.Linear(fc2, 1)
        
    def forward(self, state):
        #Policy output
        x1 = self.fc1(state)
        action_probs = self.action_layer(x1)

        #Value output
        x2 = self.fc2(state)
        state_value = self.value_layer(x2)
        
        return action_probs, state_value

    
class Agent():
    def __init__(self, input_shape, action_size, gamma, fc1, fc2):
        self.action_size = action_size
        self.input_shape = input_shape
        self.gamma = gamma
        self.network = ActorCritic(input_shape, action_size, fc1, fc2).to(device)

        self.t_step = 0
        
        self.optimizer = optim.Adam(self.network.parameters(), lr = LR, betas = betas)

        self.rewards = []
        self.log_probs = []
        self.state_values = []
    
    def get_action(self, state, sample):

        state = torch.Tensor(state).float().unsqueeze(0).to(device)
        action_probs, state_val = self.network(state)
        action_probs_dist = torch.distributions.Categorical(action_probs)
        if sample:
            action = action_probs_dist.sample()
            self.log_probs += [action_probs_dist.log_prob(action)]
            self.state_values += [state_val]
            return action.item()
        else:
            action = np.argmax(action_probs.cpu().data.numpy())
            return action
    
    def learn(self):
        self.optimizer.zero_grad()
        
        returns = []
        disc_rew = 0
        for rew in self.rewards[::-1]:
            disc_rew = rew + GAMMA*disc_rew
            returns.insert(0, disc_rew)
        
        loss = 0
        for logprob, value, total_return in zip(self.log_probs, self.state_values, returns):
            advantage = total_return  - value.item()
            actor_loss = -logprob * advantage
            critic_loss = torch.abs(value-total_return)
            loss += (actor_loss + critic_loss)

        loss.backward()
        self.optimizer.step()   
    
    def clearmemory(self):
        self.log_probs = []
        self.rewards = []
        self.state_values = []

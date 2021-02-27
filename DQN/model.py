import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, deque
import numpy as np
import random
import torch.nn.functional as F

BUFFER_SIZE = int(1e7)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
LR = 5e-4               # learning rate 
UPDATE_EVERY = 8        # how often to update the network
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ReplayBuffer:

    def __init__(self, action_size, buffer_size, batch_size):

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add_exp(self, state, action, reward, next_state, done):
        state = np.array(state)/255
        next_state = np.array(next_state)/255
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample_batch(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.Tensor(np.vstack([e.state for e in experiences if e is not None])).to(device)
        actions = torch.Tensor(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.Tensor(np.vstack([e.reward for e in experiences if e is not None])).to(device)
        next_states = torch.Tensor(np.vstack([e.next_state for e in experiences if e is not None])).to(device)
        dones = torch.Tensor(np.vstack([e.done for e in experiences if e is not None])).to(device)
        
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

class Agent():

    def __init__(self, action_size, input_shape, input_channels):
    
        self.input_shape = input_shape
        self.action_size = action_size

        self.qnetwork_local = QNetwork(action_size, input_shape, input_channels).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)
    
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        action = np.argmax(np.array(action))

        self.memory.add_exp([state], action, reward, [next_state], done)
        
        self.t_step +=1
        if self.t_step % UPDATE_EVERY == 0:
            if len(self.memory) > BATCH_SIZE:
                for i in range(5):
                    experiences = self.memory.sample_batch()
                    self.learn(experiences, GAMMA)

    def get_action(self, state, eps):
        # Epsilon-greedy action selection
        if random.random() > eps:
            state = torch.Tensor(state/255).float().unsqueeze(0).to(device)
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()
            action_idx = np.argmax(action_values.cpu().data.numpy())
            action = np.zeros((self.action_size,))
            action[action_idx] = 1
            return list(action)
        else:
            action_idx = random.choice(np.arange(self.action_size))
            action = np.zeros((self.action_size,))
            action[action_idx] = 1
            return list(action)
        

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        Q_targets_next, _ = torch.max(self.qnetwork_local(next_states).detach(), dim = 1, keepdim = True)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
    
        Q_expected = torch.gather(self.qnetwork_local(states), dim = 1, index = actions)
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


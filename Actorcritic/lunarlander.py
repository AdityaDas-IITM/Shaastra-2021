import gym
from model import Agent
import torch
import matplotlib.pyplot as plt
import time
import random

'''
state = [x, y, vx, vy, theta, vtheta, left leg on ground, right leg on ground]

actions = do nothing, fire left, fire right, fire main

Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points. 
If lander moves away from landing pad it loses reward.
Episode finishes if the lander crashes or comes to rest, receiving additional -100 or +100 points.
Each leg ground contact is +10. 
Firing main engine is -0.3 points each frame.
'''

def train(game, num_episodes, agent):
    torch.manual_seed(543)
    for k in range(num_episodes):
        done = False
        state = game.reset()
        score = 0
        while not done:
            action = agent.get_action(state, sample = True)
            next_state, reward, done, _ = game.step(action)
            agent.rewards.append(reward)
            state = next_state
            score += reward
        agent.learn()
        agent.clearmemory()
        print(f'Episode {k} Score {score}')
        if score > 200:
            torch.save(agent.network.state_dict(), "./models/LunarLander.pth")

def infer(game, vid, agent, arbit = False, path = './models/LunarLander.pth'):
    agent.clearmemory()
    if not arbit:
        agent.network.load_state_dict(torch.load(path))
    for k in range(5):
        done = False
        state = game.reset()
        while not done:
            game.render()
            #vid.capture_frame()
            if not arbit:
                action = agent.get_action(state, sample = False)
            else:
                action = random.choice(range(action_size))
            next_state, reward, done, _ = game.step(action)
            state = next_state
            time.sleep(0.015)
        





if __name__=='__main__':
    #pip install gym[Box2D] -- maybe needed
    game = gym.make('LunarLander-v2')
    #vid = gym.wrappers.monitoring.video_recorder.VideoRecorder(game, path = './random.mp4')
    action_size = game.action_space.n
    print("Action size ", action_size)

    state_size = game.observation_space.shape[0]
    print("State size ", state_size)
    num_episodes = 2000
    agent = Agent(state_size, action_size, gamma = 0.99, fc1 = 64, fc2 = 64)

    #train(game, num_episodes, agent)
    infer(game, vid, agent, arbit = False)
    #game.close()
    
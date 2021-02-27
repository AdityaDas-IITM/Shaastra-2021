import numpy as np
import random
from matplotlib import animation
import matplotlib.pyplot as plt


class SnakeGame:
    def __init__(self, board_size):
        self.h, self.w = board_size
    
    def add_frame(self, done = False):
        board = np.copy(self.board)
        if not done:
            for pos in self.snake_body:
                board[pos[1], pos[0]] = [4/255, 82/255, 217/255]
        board[self.food_pos[1], self.food_pos[0]] = [217/255, 4/255, 86/255]
        if self.save_frames:
            self.frames.append(board.tolist())
        else:
            self.frames = [board.tolist()]

    def render(self):
        a = plt.imshow(self.frames[-1], animated = True)
        plt.axis('off')
        plt.draw()
        plt.pause(0.001)
        return a

    def reset_board(self, save_frames = False):
        plt.close('all')
        self.board = np.zeros((self.h, self.w, 3))
        for i in range(self.w):
            for j in range(self.h):
                if (i+j)%2 == 0:
                    self.board[i,j] = [153/255, 217/255, 4/255]
                else:
                    self.board[i,j] = [130/255, 217/255, 4/255]
        self.snake_body = []
        self.head_x, self.head_y = random.choice(range(self.w)), random.choice(range(self.h))
        self.snake_body.append([self.head_x, self.head_y])
        self.pellets = 0
        self.placefood()
        self.frames = []
        self.save_frames = save_frames
        self.add_frame()
        state = self.get_obs()
        return state

    def step(self, action):
        # 0:left, 1:right, 2:up, 3:down
        curr_state = self.get_obs()
        done = False
        if action == 0:
            self.head_x -= 1
        elif action == 1:
            self.head_x += 1
        elif action == 2:
            self.head_y -= 1
        elif action == 3:
            self.head_y += 1
        
        self.snake_body.insert(0, [self.head_x, self.head_y])

        if self.head_x < 0 or self.head_x >= self.w or self.head_y < 0 or self.head_y >= self.h: #corssed boundary
            done = True
            state = np.zeros(8,)
            reward = -50

        if [self.head_x, self.head_y] in self.snake_body[1:]: # ate itself
            done = True
            state = np.zeros(8,)
            reward = -50

        if self.head_x == self.food_pos[0] and self.head_y == self.food_pos[1]: # ate food
            reward = 100
            self.pellets+=1
            self.placefood()
            state = self.get_obs()
        elif curr_state[action] == 1: # moved towards food
            reward = 10
            self.snake_body = self.snake_body[:-1]
            state = self.get_obs()
        else:                         # did not move towards food
            reward = -10
            self.snake_body = self.snake_body[:-1]
            state = self.get_obs()

        self.add_frame(done)
        
        return state, reward, done

    def placefood(self):
        self.food_pos = [random.choice(range(self.w)), random.choice(range(self.h))]
        while self.food_pos in self.snake_body:
            self.food_pos = [random.choice(range(self.w)), random.choice(range(self.h))]
        
    def get_obs(self):
        food_state = [0, 0, 0, 0] # left, right, up, down
        if self.food_pos[0] - self.head_x < 0:
            food_state[0]  = 1
        if self.food_pos[0] - self.head_x > 0:
            food_state[1]  = 1
        if self.food_pos[1] - self.head_y < 0:
            food_state[2]  = 1
        if self.food_pos[1] - self.head_y > 0:
            food_state[3]  = 1
        
        obs_state = [0, 0, 0, 0] # left, right, up, down
        if self.head_x==0 or [self.head_x-1, self.head_y] in self.snake_body[1:]:
            obs_state[0] = 1
        if self.head_x==self.w or [self.head_x+1, self.head_y] in self.snake_body[1:]:
            obs_state[1] = 1
        if self.head_y==0 or [self.head_x, self.head_y-1] in self.snake_body[1:]:
            obs_state[2] = 1
        if self.head_y==self.h or [self.head_x, self.head_y+1] in self.snake_body[1:]:
            obs_state[3] = 1
        
        return food_state+obs_state
    
    def create_vid(self, path):
        fig = plt.figure()
        ims = []
        for frame in self.frames:
            im = plt.imshow(frame, animated = True)
            ims.append([im])
        anim = animation.ArtistAnimation(fig, ims, interval = 100, repeat = False, blit = False)
        anim.save(path)

if __name__ == '__main__':
    game = SnakeGame((30,30))
    action_dict = {'L':0, 'R':1, 'U':2, 'D':3}
    state = game.reset_board()
    print(state)
    
    fig = plt.figure()
    done = False
    while not done:
        a = game.render()
        action_str = input("Action: ")
        action = action_dict[action_str]
        state, reward, done = game.step(action)
        print(state, reward, done)
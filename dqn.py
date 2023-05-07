#!/usr/bin/env python3
"""
Brandon L Morris

Adapted from work by E. Culurciello
"""

from absl import app, flags
import itertools
from random import sample, randint, random
import numpy as np
from skimage.transform import resize
from time import time, sleep
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import trange
from vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution
import matplotlib.pyplot as plt 


# Q-learning settings
learning_rate = 0.00025
discount = 0.99
epochs = 10
iters = 2000

# NN learning settings
batch_size = 64
save_model = True
load_model = False
skip_training = False
watch_episodes = 10
save_path = 'model-doom.pth'
# Default configuration file path
default_config_file_path = "/home/jagadeesh/ViZDoom/scenarios/deathmatch.cfg"
config = default_config_file_path

replay_memory = 10000
# Training regime
test_episodes = 100

frame_repeat = 12
resolution = (30, 45)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def preprocess(img):
    return torch.from_numpy(resize(img, resolution).astype(np.float32))


def game_state(game):
    return preprocess(game.get_state().screen_buffer)


class ReplayMemory:
    def __init__(self, capacity):
        channels = 1
        state_shape = (capacity, channels, *resolution)
        self.s1 = torch.zeros(state_shape, dtype=torch.float32).to(device)
        self.s2 = torch.zeros(state_shape, dtype=torch.float32).to(device)
        self.a = torch.zeros(capacity, dtype=torch.long).to(device)
        self.r = torch.zeros(capacity, dtype=torch.float32).to(device)
        self.isterminal = torch.zeros(capacity, dtype=torch.float32).to(device)

        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def add_transition(self, s1, action, s2, isterminal, reward):
        idx = self.pos
        self.s1[idx,0,:,:] = s1
        self.a[idx] = action
        if not isterminal:
            self.s2[idx,0,:,:] = s2
        self.isterminal[idx] = isterminal
        self.r[idx] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, size):
        idx = sample(range(0, self.size), size)
        return (self.s1[idx], self.a[idx], self.s2[idx], self.isterminal[idx],
                self.r[idx])


class QNet(nn.Module):
    def __init__(self, available_actions_count):
        super(QNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=6, stride=3) # 8x9x14
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=2) # 8x4x6 = 192
        self.fc1 = nn.Linear(192, 128)
        self.fc2 = nn.Linear(128, available_actions_count)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), learning_rate)
        self.memory = ReplayMemory(capacity=replay_memory)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 192)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def get_best_action(self, state):
        q = self(state)
        _, index = torch.max(q, 1)
        return index

    def train_step(self, s1, target_q):
        output = self(s1)
        loss = self.criterion(output, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def learn_from_memory(self):
        if self.memory.size < batch_size: return
        s1, a, s2, isterminal, r = self.memory.get_sample(batch_size)
        q = self(s2).detach()
        q2, _ = torch.max(q, dim=1)
        target_q = self(s1).detach()
        idxs = (torch.arange(target_q.shape[0]), a)
        target_q[idxs] = r + discount * (1-isterminal) * q2
        self.train_step(s1, target_q)


def find_eps(epoch):
    """Balance exploration and exploitation as we keep learning"""
    start, end = 1.0, 0.1
    const_epochs, decay_epochs = .1*epochs, .6*epochs
    if epoch < const_epochs:
        return start
    elif epoch > decay_epochs:
        return end
    # Linear decay
    progress = (epoch-const_epochs)/(decay_epochs-const_epochs)
    return start - progress * (start - end)


def perform_learning_step(epoch, game, model, actions):
    s1 = game_state(game)
    if random() <= find_eps(epoch):
        a = torch.tensor(randint(0, len(actions) - 1)).long()
    else:
        s1 = s1.reshape([1, 1, *resolution])
        a = model.get_best_action(s1.to(device))
    reward = game.make_action(actions[a], frame_repeat)

    if game.is_episode_finished():
        isterminal, s2 = 1., None
    else:
        isterminal = 0.
        s2 = game_state(game)

    model.memory.add_transition(s1, a, s2, isterminal, reward)
    model.learn_from_memory()


def initialize_vizdoom(config):
    game = DoomGame()
    game.load_config(config)
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.GRAY8)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.init()
    return game


def train(game, model, actions):
    time_start = time()
    total_training_rewards = []
    print("Saving the network weigths to:", save_path)
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}')
        episodes_finished = 0
        scores = np.array([])
        game.new_episode()
        for learning_step in trange(iters, leave=False):
            perform_learning_step(epoch, game, model, actions)
            if game.is_episode_finished():
                score = game.get_total_reward()
                total_training_rewards.append(score)
                scores = np.append(scores, score)
                game.new_episode()
                episodes_finished += 1
        print(f'Completed {episodes_finished} episodes')
        print(f'Mean: {scores.mean():.1f} +/- {scores.std():.1f}')
        print("Testing...")
        test(test_episodes, game, model, actions)
        torch.save(model, save_path)
    make_image(total_training_rewards,"qlearning_deathmatch.png",window=25)
    print(f'Total elapsed time: {(time()-time_start):.2f} minutes')


def test(iters, game, model, actions):
    scores = np.array([])
    for _ in trange(test_episodes, leave=False):
        game.new_episode()
        while not game.is_episode_finished():
            state = game_state(game)
            state = state.reshape([1, 1, resolution[0], resolution[1]])
            a_idx = model.get_best_action(state.to(device))
            game.make_action(actions[a_idx], frame_repeat)
        r = game.get_total_reward()
        scores = np.append(scores, r)
    print(f'Results: mean: {scores.mean():.1f} +/- {scores.std():.1f}')

def make_image(scores, filename, x=None, window=5):   
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
    if x is None:
        x = [i for i in range(N)]
    plt.ylabel('Score')       
    plt.xlabel('Game')                     
    plt.plot(x, running_avg)
    plt.savefig(filename)

import cv2
fps = 24

out = cv2.VideoWriter('deathmatchq.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (640, 480), False)
def watch_episodes(game, model, actions):
    game.set_window_visible(True)
    game.set_mode(Mode.ASYNC_PLAYER)
    game.init()
    for episode in range(watch_episodes):
        game.new_episode(f'episode-{episode}')
        while not game.is_episode_finished():
            out.write(game.get_state().screen_buffer)
            state = game_state(game)
            state = state.reshape([1, 1, resolution[0], resolution[1]])
            a_idx = model.get_best_action(state.to(device))
            game.set_action(actions[a_idx])
            for _ in range(frame_repeat):
                game.advance_action()
        sleep(1.0)
        score = game.get_total_reward()
        print(f'Total score: {score}')
    
    out.release()

def main(_):
    game = initialize_vizdoom(config)

    n = game.get_available_buttons_size()
    actions = [list(a) for a in itertools.product([0, 1], repeat=n)]

    if load_model:
        print(f'Loading model from: {save_path}')
        model = torch.load(save_path).to(device)
    else:
        model = QNet(len(actions)).to(device)

    print("Starting the training!")
    if not skip_training:
        train(game, model, actions)

    game.close()
    print("======================================")
    watch_episodes(game, model, actions)


if __name__ == '__main__':
    app.run(main)


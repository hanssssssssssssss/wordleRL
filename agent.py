# import torch
import random
import numpy as np
from collections import deque

import torch

from wordle import Wordle
from model import Linear_QNet, QTrainer

MAX_ROUNDS = 6
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001


class Agent:
    def __init__(self, action_space_len, gamma, epsilon):
        self.n_games = 0
        self.action_space_len = action_space_len
        self.gamma = gamma  # TODO: what is this??
        self.epsilon = epsilon
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Linear_QNet(input_size=391,
                                 hidden_size=256,
                                 output_size=action_space_len)
        self.trainer = QTrainer(self.model,
                                learning_rate=LEARNING_RATE,
                                gamma=self.gamma)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_prediction_index(self, state):
        if random.randint(0, self.epsilon) < (self.epsilon - self.n_games):
            prediction_index = random.randint(0, self.action_space_len)
        else:
            # only exploitation no exploration so far
            state0 = torch.tensor(state, dtype=torch.float)
            predicted_probabilities = self.model(state0)
            prediction_index = torch.argmax(predicted_probabilities).item()
        return prediction_index


def word_one_hot(word):
    """one hot encode a word for each of the 26 letters at each of the 5 position"""
    encoding = np.zeros(26 * 5)
    for i, letter in enumerate(word):
        encoding[i * 26 + (ord(letter) - 97)] = 1
        encoding[i * 26 + (ord(letter) - 97)] = 1
    return encoding


def train(random_seed=None, vocab_subset_len=None):
    total_games = 0
    wins = 0
    max_wins = 0
    with open("data/possible_words.txt") as word_list:
        vocab = word_list.readlines()
    if vocab_subset_len:
        random.seed(random_seed)
        vocab = random.sample(vocab, vocab_subset_len)
    random.seed(random_seed)
    solution = random.choice(vocab)
    game = Wordle(vocab, MAX_ROUNDS, solution)
    agent = Agent(gamma=.9,
                  epsilon=100,
                  action_space_len=len(vocab))
    while True:
        state_old = game.state
        prediction_index = agent.get_prediction_index(state_old)
        word = vocab[prediction_index]
        state_new, reward = game.set_state(word)

        agent.train_short_memory(state_old, word, reward, state_new, game.over)
        agent.remember(state_old, word, reward, state_new, game.over)

        if game.over:
            total_games += 1
            agent.train_long_memory()
            if game.won:
                wins += 1
            if (total_games % 1000) == 0:
                print("Played: {} Won:{} of the last 1000".format(
                    total_games, wins))
                if wins > max_wins:
                    max_wins = wins
                    agent.model.save()
                wins = 0
            game = Wordle(vocab, MAX_ROUNDS, solution)


if __name__ == '__main__':
    train()

# import torch
import random
import numpy as np
from collections import deque
from wordle import Wordle
from model import Linear_QNet, QTrainer

MAX_ROUNDS = 6
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001


class Agent:
    def __init__(self):
        self.n_games = 0
        self.gamma = 0.9  # TODO: what is this??
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Linear_QNet(input_size=391,
                                 hidden_size=256,
                                 output_size=130)
        self.trainer = QTrainer(self.model,
                                learning_rate=LEARNING_RATE,
                                gamma=self.gamma)

    def remember(self, state, action, reward, next_state, won):
        self.memory.append((state, action, reward, next_state, won))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, wons = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, wons)

    def train_short_memory(self, state, action, reward, next_state, won):
        self.trainer.train_step(state, action, reward, next_state, won)


def word_one_hot(word):
    """one hot encode a word for each of the 26 letters at each of the 5 position"""
    encoding = np.zeros(26 * 5)
    for i, letter in enumerate(word):
        encoding[i * 26 + (ord(letter) - 97)] = 1
    return encoding


def train():
    total_games = 0
    won_games = 0
    agent = Agent()
    vocab = []
    vocab_one_hot_matrix = np.zeros((len(vocab), 130))
    with open("data/possible_words.txt") as word_list:
        for i, word in enumerate(word_list):
            vocab.append(word)
            vocab_one_hot_matrix[i] = word_one_hot(word)
    solution = ""  # TODO: random or hardcoded solution
    game = Wordle(vocab, MAX_ROUNDS, solution)
    while True:
        state_old = game.state
        prediction_probabilities = np.dot(vocab_one_hot_matrix, state_old)
        word = random.choices(vocab, prediction_probabilities)
        state_new, reward = game.set_state(word)

        agent.train_short_memory(state_old, word, reward, state_new, game.won)
        agent.remember(state_old, word, reward, state_new, game.won)

        if game.over:
            total_games += 1
            agent.train_long_memory()
            if game.won:
                won_games += 1
            if (total_games % 1000) == 0:
                print("Played: {} Won:{} of the last 1000".format(
                    total_games, won_games))
                won_games = 0
            game = Wordle(vocab, MAX_ROUNDS, solution)


if __name__ == '__main__':
    train()
    
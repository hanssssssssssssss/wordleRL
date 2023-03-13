import sys
import random
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
        self.n_wins = 0
        self.action_space_len = action_space_len
        self.gamma = gamma  # TODO: what is this??
        self.epsilon = epsilon
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Linear_QNet(input_size=390,
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
        random.seed(None)
        if random.randint(0, self.epsilon) < (self.epsilon - self.n_games):
            random.seed(None)
            prediction_index = random.randint(0, self.action_space_len - 1)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            predicted_probabilities = self.model(state0)
            prediction_index = torch.argmax(predicted_probabilities).item()
        return prediction_index


def train(vocab_subset_len=None, random_seed=None):
    recent_wins = 0
    max_wins = 0
    with open("data/possible_words.txt") as word_list:
        vocab = word_list.read().split('\n')
    if vocab_subset_len:
        random.seed(random_seed)
        vocab = random.sample(vocab, k=vocab_subset_len)
    random.seed(random_seed)
    solution = random.choice(vocab)
    game = Wordle(vocab, MAX_ROUNDS, solution)
    agent = Agent(gamma=.9,
                  epsilon=1000,
                  action_space_len=len(vocab))

    while True:
        state_old = game.state
        prediction_index = agent.get_prediction_index(state_old)
        word = vocab[prediction_index]
        state_new, reward = game.set_state(word)

        agent.train_short_memory(state_old, prediction_index, reward, state_new, game.over)

        if game.over:
            agent.n_games += 1
            if len(agent.memory) > 0:
                agent.train_long_memory()
            if game.won:
                recent_wins += 1
                agent.n_wins += 1
                agent.remember(state_old, prediction_index, reward, state_new, game.over)
            if (agent.n_games % 100) == 0:
                print("Played: {} Recently won: {}/100 Total win rate: {} ".format(
                    agent.n_games, recent_wins, agent.n_wins/agent.n_games))
                if recent_wins > max_wins:
                    max_wins = recent_wins
                    agent.model.save()
                recent_wins = 0
            random.seed(random_seed)
            solution = random.choice(vocab)
            game = Wordle(vocab, MAX_ROUNDS, solution)


if __name__ == '__main__':
    train(vocab_subset_len=int(sys.argv[1]) if len(sys.argv) > 1 else None,
          random_seed=int(sys.argv[2]) if len(sys.argv) > 2 else None)

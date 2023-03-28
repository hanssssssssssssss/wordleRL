import sys
import signal
import random
from pprint import pprint
from collections import deque, Counter
import torch
from wordle import Wordle
from model import Linear_QNet, QTrainer

MAX_MEMORY = 10_000
BATCH_SIZE = 1000
LEARNING_RATE = .001
GAMMA = .8
EPSILON = 1000

class Agent:
    def __init__(self, gamma, epsilon, vocab):
        self.green = []
        self.yellow = []
        self.grey = []
        self.n_games = 0
        self.n_wins = 0
        self.action_space_len = len(vocab)
        self.gamma = gamma  # TODO: what is this??
        self.epsilon = epsilon
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(input_size=390,
                                 hidden_size=256,
                                 output_size=130,
                                 vocab=vocab)
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

    def get_prediction_baseline(self, state, last_guess, vocab):
        if not last_guess:
            return random.choice(vocab)
        else:
            for i, char in enumerate(last_guess):
                position_offset = i * 26 * 3
                char_offset = (ord(char) - 97) * 3
                position_state = state[position_offset + char_offset:position_offset + char_offset + 3]
                #print(char, position_state)
                if position_state[0]:
                    self.grey.append(char)
                elif position_state[1]:
                    self.yellow.append(char)
                elif position_state[2]:
                    self.green.append(char)
                if char in self.grey and (char in self.yellow or char in self.green):
                    self.grey.remove(char)
            for word in vocab:
                green = list.copy(self.green)
                yellow = list.copy(self.yellow.copy())
                dismissed = (word == last_guess)
                if not dismissed:
                    for i, char in enumerate(word):
                        if char in self.grey:
                            dismissed = True
                            break
                        elif char in green:
                            if char != last_guess[i]:
                                #dismissed = True
                                break
                            else:
                                green.remove(char)
                        elif char in yellow:
                            yellow.remove(char)
                if not dismissed:
                    return word


def train(vocab_subset_len=None, random_seed=None, n_solutions=None, saved_model_path=None):
    recent_wins = 0
    max_wins = 0
    with open("data/possible_words.txt") as word_list:
        vocab = word_list.read().split('\n')
    if vocab_subset_len:
        random.seed(random_seed)
        vocab = random.sample(vocab, k=vocab_subset_len)
    if n_solutions:
        random.seed(random_seed)
        solutions = random.sample(vocab, k=n_solutions)
    else:
        solutions = vocab
    guessed_words_counter = Counter(vocab)
    winners_counter = Counter(vocab)
    guessed_words_indices = []

    def keyboard_interrupt(sig, frame):
        print("Guesses:")
        pprint(guessed_words_counter)
        print("Solved:")
        pprint(winners_counter)
        sys.exit(0)
    signal.signal(signal.SIGINT, keyboard_interrupt)

    game = Wordle(vocab, random.choice(solutions))
    agent = Agent(gamma=GAMMA,
                  epsilon=EPSILON,
                  vocab=vocab)
    if saved_model_path:
        agent.model.load(saved_model_path)

    while True:
        state_old = game.state
        prediction_index = agent.get_prediction_index(state_old)
        word = vocab[prediction_index]
        guessed_words_counter[word] += 1
        state_new, reward = game.set_state(word)
        guessed_words_indices.append(prediction_index)
        agent.train_short_memory(state_old.copy(), prediction_index, reward, state_new.copy(), game.over)
        agent.remember(state_old.copy(), prediction_index, reward, state_new.copy(), game.over)

        if game.over:
            guessed_words_indices = []
            agent.n_games += 1
            if len(agent.memory) > 0:
                agent.train_long_memory()
            if game.won:
                recent_wins += 1
                agent.n_wins += 1
                winners_counter[game.solution] += 1
            if (agent.n_games % 100) == 0:
                print("Played: {} Recently won: {}/100 Total win rate: {:.4f}".format(
                    agent.n_games, recent_wins, agent.n_wins/agent.n_games))
                if recent_wins > max_wins:
                    max_wins = recent_wins
                    agent.model.save()
                recent_wins = 0
            game = Wordle(vocab, random.choice(solutions))


def train_baseline(vocab_subset_len=None, random_seed=None, n_solutions=None, saved_model_path=None):
    recent_wins = 0
    max_wins = 0
    with open("data/possible_words.txt") as word_list:
        vocab = word_list.read().split('\n')
    if vocab_subset_len:
        random.seed(random_seed)
        vocab = random.sample(vocab, k=vocab_subset_len)
    if n_solutions:
        random.seed(random_seed)
        solutions = random.sample(vocab, k=n_solutions)
    else:
        solutions = vocab
    guessed_words_counter = Counter(vocab)
    winners_counter = Counter(vocab)
    def keyboard_interrupt(sig, frame):
        print("Guesses:")
        pprint(guessed_words_counter)
        print("Solved:")
        pprint(winners_counter)
        sys.exit(0)
    signal.signal(signal.SIGINT, keyboard_interrupt)

    game = Wordle(vocab, random.choice(solutions))

    prev_guess = None
    agent = Agent(gamma=GAMMA,
                  epsilon=EPSILON,
                  vocab=vocab)
    while True:
        state_old = game.state
        word = agent.get_prediction_baseline(state_old, prev_guess, vocab)
        prev_guess = word
        #print(agent.grey, agent.yellow, agent.green)
        #print(word, game.solution)
        guessed_words_counter[word] += 1
        game.set_state(word)
        if game.over:
            agent.green = []
            agent.grey = []
            agent.yellow = []
            agent.n_games += 1
            if len(agent.memory) > 0:
                agent.train_long_memory()
            if game.won:
                recent_wins += 1
                agent.n_wins += 1
                winners_counter[game.solution] += 1
            if (agent.n_games % 100) == 0:
                print("Played: {} Recently won: {}/100 Total win rate: {:.4f}".format(
                    agent.n_games, recent_wins, agent.n_wins/agent.n_games))
                if recent_wins > max_wins:
                    max_wins = recent_wins
                    agent.model.save()
                recent_wins = 0
            game = Wordle(vocab, random.choice(solutions))




if __name__ == '__main__':
    train_baseline(vocab_subset_len=int(sys.argv[1]) if len(sys.argv) > 1 else None,
          random_seed=int(sys.argv[2]) if len(sys.argv) > 2 else None,
          n_solutions=int(sys.argv[3]) if len(sys.argv) > 3 else None,
          saved_model_path=sys.argv[4] if len(sys.argv) > 4 else None)

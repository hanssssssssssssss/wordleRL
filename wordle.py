import collections
import numpy as np


class Wordle:
    def __init__(self, vocab, max_rounds, solution):
        self.vocab = vocab
        self.max_rounds = max_rounds
        self.round = 0
        self.solution = solution
        self.state = np.zeros(390, dtype=int)
        self.won = False
        self.over = False

    def guess(self, word):
        if self.over:
            return False
        else:
            self.set_state(word.lower())
            return True

    def play_visual(self):
        while True:
            guess = input()
            if guess == "q":
                return
            if len(guess) != 5:
                print("Word must be 5 characters long")
            # TODO: check non ASCII chars
            else:
                if self.guess(guess):
                    self._visualize_state(self.state, guess)
                if self.over:
                    print("Game over, you ", "won!" if self.won else "lost!")
                    return

    def set_state(self, word):
        self.round += 1
        reward = 0
        solution_char_count = collections.Counter(self.solution)
        seen_char_count = collections.Counter()
        for i, char in enumerate(word):
            position_offset = i * 26 * 3
            char_offset = (ord(char) - 97) * 3
            if solution_char_count[char] > 0:
                # Character exists in solution
                if word[i] == self.solution[i]:
                    # Character is correct at this position
                    # Set "definitely" on this position (and reset "maybe")
                    self.state[position_offset + char_offset:position_offset + char_offset + 3] = [0, 0, 1]
                    reward += 5
                    if solution_char_count[char] - seen_char_count[char] == 0:
                        # Character has been seen more often than it exists
                        # Reset last seen maybe to "definitely not"
                        for n in range(i):
                            prev_position_offset = (i - (n + 1)) * 26 * 3
                            prev_position_state = self.state[prev_position_offset + char_offset + 1]
                            if prev_position_state == 1:
                                self.state[prev_position_offset + char_offset:
                                           prev_position_offset + char_offset + 3] = [1, 0, 0]
                                break
                elif solution_char_count[char] - seen_char_count[char] > 0:
                    # Character has been seen less often than it exists
                    # Set "maybe" on this position
                    self.state[position_offset + char_offset + 1] = 1
                    reward += 1
                else:
                    # Character has been seen more often than it exists
                    # Reset "maybe" on this position to "definitely not"
                    self.state[position_offset + char_offset:
                              position_offset + char_offset + 3] = [1, 0, 0]
            else:
                if char in self.solution:
                    # character exists in solution but has already been seen as often as it exists
                    # Set "definitely not" on this position, reset "maybe" for all remaining positions
                    self.state[position_offset + char_offset:position_offset + char_offset + 3] = [1, 0, 0]
                    for n in range(i, len(word)):
                        other_positions_offset = n * 26 * 3
                        self.state[other_positions_offset + char_offset + 1] = 0
                else:
                    # character does not exist in solution
                    # Set "definitely not" on all positions
                    for n in range(len(word)):
                        other_positions_offset = n * 26 * 3
                        self.state[other_positions_offset + char_offset:
                                   other_positions_offset + char_offset + 3] = [1, 0, 0]

            seen_char_count[char] += 1

        if word == self.solution:
            self.won = True
            reward += 50
            self.over = True
        elif self.round > self.max_rounds:
            reward = -50
            self.over = True
        return self.state, reward

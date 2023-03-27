import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class   Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, vocab):
        super().__init__()
        self.f0 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

        vocab_one_hot = np.zeros((len(vocab), 130))
        for i, word in enumerate(vocab):
            for j, char in enumerate(word):
                vocab_one_hot[i, j*26 + (ord(char) - 97)] = 1
        self.words_one_hot = torch.Tensor(vocab_one_hot)

    def forward(self, x):
        y = self.f0(x.float())
        return torch.tensordot(y, self.words_one_hot, dims=((-1,), (1,)))

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class QTrainer:
    def __init__(self, model, learning_rate, gamma):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(),
                                    lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(action, dtype=torch.int)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        prediction = self.model(state)
        target = prediction.clone()
        for i in range(len(done)):
            Q_new = reward[i]
            if not done[i]:
                Q_new = reward[i] + self.gamma * torch.max(self.model(next_state[i]))
            target[i][action[i].item()] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, prediction)
        loss.backward()
        self.optimizer.step()

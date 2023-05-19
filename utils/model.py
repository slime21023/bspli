import torch
import torch.nn as nn
from torch import optim

# check if the GPU runtime env is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MLP():
    def __init__(self, input_size, num_classes, hidden_size=100):
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2),
             nn.Linear(hidden_size, num_classes)
        ).to(device)

    def train(self, X, y):
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        for t in range(2000):
            y_pred = self.model(X)

            loss = self.loss_fn(y_pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()



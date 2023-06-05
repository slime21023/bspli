import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

# check if the GPU runtime env is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MLP():
    def __init__(self, input_size, num_classes, epoch_num = 5, error_rate=0.03, hidden_size=100):
        self.error_rate = error_rate
        self.epoch_num = epoch_num
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, int(hidden_size/2)),
            nn.LeakyReLU(0.2),
            nn.Linear(int(hidden_size/2), 1),
        ).to(device)

    def train(self, train_loader):
        # Define your execution device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Convert model parameters and buffers to CPU or Cuda
        self.model.to(device)

        # self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        self.optimizer = optim.Adamax(self.model.parameters(), lr=0.1)
        self.loss_fn = nn.SmoothL1Loss(reduction ="mean")
        self.model.train()

        for epoch  in range(self.epoch_num):
            for batch, (points, labels) in enumerate(train_loader, 0):
                labels = labels.reshape(labels.shape[0])

                # get the inputs
                points = Variable(points.type(torch.FloatTensor))
                labels = Variable(labels.type(torch.FloatTensor))

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # predict classes using points from the training set
                outputs = self.model(points.to(torch.float32))

                # compute the loss based on model output and real labels
                loss = self.loss_fn(outputs, labels)

                # backpropagate the loss
                loss.backward()

                # adjust parameters based on the calculated gradients
                self.optimizer.step()

                if batch % 100 == 99:
                    print(f'{epoch + 1}, {batch + 1} loss: {loss.item() / 100 }')

        self.model.eval()

    def pred(self, X):
        self.model.eval()
        return self.model(X)




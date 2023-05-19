import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

# check if the GPU runtime env is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MLP():
    def __init__(self, input_size, num_classes, error_rate=0.03, hidden_size=100):
        self.error_rate = error_rate
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, num_classes),
            nn.Softmax(dim=1)
        ).to(device)

    def train(self, train_loader):
        # Define your execution device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Convert model parameters and buffers to CPU or Cuda
        self.model.to(device)

        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        self.loss_fn = torch.nn.NLLLoss()
        self.model.train()

        for epoch  in range(2000):
            running_loss = 0.0
            for batch, (points, labels) in enumerate(train_loader, 0):
            
                # get the inputs
                points = Variable(points.to(torch.float32))
                labels = Variable(labels.type(torch.float32))
                print(f"points shape: {points.shape}")
                print(f"labels shape: {labels.shape}")

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # predict classes using points from the training set
                outputs = self.model(points)
                print(f"outputs shape: {outputs.shape}")

                # compute the loss based on model output and real labels
                loss = self.loss_fn(outputs, labels)

                # backpropagate the loss
                loss.backward()

                # adjust parameters based on the calculated gradients
                self.optimizer.step()

                running_loss += loss.item()     # extract the loss value
                if batch % 100 == 99:
                    # print every 1000 (twice per epoch) 
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, batch + 1, running_loss / 1000))
                    
                    # zero the loss
                    running_loss = 0.0
        self.model.eval()

    def pred(self, X):
        self.model.eval()
        y_hat = torch.max(self.model(X), 1)
        return y_hat




import torch
from torch import nn
from torch.nn import functional as F


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(21120, 1024)
        self.fc2 = nn.Linear(1024, 36)
        # self.conv1 = nn.Conv2d(3, 10, kernel_size=3)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        # self.conv2_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(20*14*5, 50)
        # self.fc2 = nn.Linear(50, 36)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        # print(x.shape)

        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))        # x = x.view(-1, 20*14*5)
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)
        #
        # return F.log_softmax(x, dim=-1)

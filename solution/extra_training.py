import torch
from torch.utils.data import dataloader
from torchvision import transforms
import os
from tqdm import tqdm
from sklearn import preprocessing
from sklearn import metrics
from MyDataset import MyDataset
from model import NeuralNet
import numpy as np
from torch.nn import functional as F


n_epochs = 5
log_interval = 32
model = NeuralNet()
model.load_state_dict(torch.load('model.pth'))
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
train_path = '../sample-code/1char_big_train/'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

train_image_labels = [f[0] for f in os.listdir(train_path) if os.path.isfile(os.path.join(train_path, f))]

train_labels_enc = [ord(label) for label in train_image_labels]
for i in range(len(train_labels_enc)):
    if train_labels_enc[i] < 58:
        train_labels_enc[i] -= 48
    else:
        train_labels_enc[i] -= 65

train_labels_enc = np.asarray(train_labels_enc)
train_labels_enc = torch.from_numpy(train_labels_enc)

data_transforms = transforms.ToTensor()
# data_transforms = None
train_dataset = MyDataset(train_path, Transform=data_transforms, labels=train_labels_enc)
train_dataloader = dataloader.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
model.to(device)

def main():
    for epoch in range(n_epochs):
        print(f"epoch:{epoch}")
        model.train()
        for batch_idx, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            output = model(data['image'].to(device))
            labels = data['label'].to(device)
            loss = F.nll_loss(output, labels)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data['image']), len(train_dataloader.dataset),
                           100. * batch_idx / len(train_dataloader), loss.item()))

            torch.save(model.state_dict(), 'model.pth')
            torch.save(optimizer.state_dict(), 'optimizer.pth')

if __name__ == '__main__':
    main()

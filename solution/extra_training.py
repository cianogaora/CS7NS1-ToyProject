import torch
from torch.utils.data import dataloader
from torchvision import transforms
import os
from MyDataset import MyDataset
from model import NeuralNet
import numpy as np
from torch.nn import functional as F


n_epochs = 20
log_interval = 64
model = NeuralNet()
model.load_state_dict(torch.load('models/model3_softmax.pth'))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# optimizer.load_state_dict(torch.load('optimizer.pth'))
train_path = '../sample-code/1char_big_train/'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(device)

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
train_dataset = MyDataset(train_path, Transform=data_transforms, labels=train_labels_enc)
train_dataloader = dataloader.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
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

            torch.save(model.state_dict(), 'models/model3_softmax.pth')
            torch.save(optimizer.state_dict(), 'optimizer.pth')

if __name__ == '__main__':
    main()

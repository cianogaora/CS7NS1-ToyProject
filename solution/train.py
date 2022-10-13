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

captcha_len = 4
num_symbols = 36

# labels_chars = [[char for char in label] for label in labels]
#             labels_flat = [c for clist in labels for c in clist]
#             lbl_enc = preprocessing.LabelEncoder()
#             lbl_enc.fit(labels_flat)
#             labels_enc = [lbl_enc.transform(x) for x in labels_chars]
#             labels_enc = np.array(labels_enc)
#             labels_enc = labels_enc + 1
#             labels_enc = torch.from_numpy(labels_enc)

def main():
    train_path = '../sample-code/1char_big_train/'
    test_path = '../sample-code/1char_test/'
    train_image_labels = [f[0] for f in os.listdir(train_path) if os.path.isfile(os.path.join(train_path, f))]
    test_image_labels = [f[0] for f in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, f))]
    print(train_image_labels[0])
    symbols = open('../sample-code/symbols.txt').readlines()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
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
    test_dataset = MyDataset(test_path, Transform=data_transforms, labels=test_image_labels)

    train_dataloader = dataloader.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = dataloader.DataLoader(test_dataset, batch_size=4, shuffle=True)

    model = NeuralNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer.load_state_dict(torch.load('optimizer.pth'))

    loss_fn = torch.nn.CrossEntropyLoss()

    running_loss = 0.
    last_loss = 0.

    train_losses = []
    train_counter = []
    n_epochs = 20
    log_interval = 10
    model.to(device)
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
                    epoch, batch_idx * len(data), len(train_dataloader.dataset),
                           100. * batch_idx / len(train_dataloader), loss.item()))

            torch.save(model.state_dict(), 'model.pth')
            torch.save(optimizer.state_dict(), 'optimizer.pth')

    path = '/Users/cian/Google Drive/Engineering/5th Year/Scalable/ToyProject/solution/model.pth'
    torch.save(model.state_dict(), path)

if __name__ == '__main__':
    main()

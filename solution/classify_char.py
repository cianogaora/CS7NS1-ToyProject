from torch.utils.data import dataloader
from model import NeuralNet
import numpy as np
import torch
from torch.nn import functional as F
from MyDataset import MyDataset
import os
from torchvision import transforms

test_path = '../sample-code/1char_test/'
test_image_labels = [f[0] for f in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, f))]

test_labels_enc = [ord(label) for label in test_image_labels]

for i in range(len(test_labels_enc)):
    if test_labels_enc[i] < 58:
        test_labels_enc[i] -= 48
    else:
        test_labels_enc[i] -= 65

test_labels_enc = np.asarray(test_labels_enc)
test_labels_enc = torch.from_numpy(test_labels_enc)

data_transforms = transforms.ToTensor()

test_dataset = MyDataset(test_path, Transform=data_transforms, labels=test_labels_enc)
test_dataloader = dataloader.DataLoader(test_dataset, batch_size=4, shuffle=True)
test_losses = []


def test(model):
    model.eval()
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for idx, data in enumerate(test_dataloader):
                output = model(data['image'])
                target = data['label']
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(test_dataloader.dataset)

            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_dataloader.dataset),
                100. * correct / len(test_dataloader.dataset)))


def main():
    network = NeuralNet()
    network.load_state_dict(torch.load('models/model3_softmax.pth'))
    test(network)

if __name__ == "__main__":
    main()

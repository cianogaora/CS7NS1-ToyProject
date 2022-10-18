from model import NeuralNet
import torch
from torchvision import transforms
import torch.nn.functional as F

import os
import cv2


def decode(num):
    if num < 10:
        num += 48
    else:
        num += 55
    return chr(num)

class Solver:
    def __init__(self, model_path, test_img_path):
        self.model = NeuralNet()
        self.model.load_state_dict(torch.load(model_path))
        self.imgs = [cv2.imread(test_img_path + img) for img in os.listdir(test_img_path)]
        for i in range(len(self.imgs)):
            self.imgs[i] = cv2.cvtColor(self.imgs[i], cv2.COLOR_BGR2GRAY)
        self.predictions = []

    def solve(self):
        imgs = self.imgs
        model = self.model
        model.eval()
        predictions = []
        image = imgs[5]
        # cv2.imshow('img', img)

        width = 26
        window = image[:, :width]
        # window = np.zeros((1, 1, 64, width))
        # # window.resize((1, 1, 64, 26))

        # window = window.astype(np.double)
        # window = torch.tensor(window).float()

        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow('win', window)
        # window[0][0][:][:] = image
        # window = torch.from_numpy(window).float()
        trans = transforms.Compose([
            transforms.ToTensor()])
        window = trans(window)
        window = window.unsqueeze(0)
        pred = model(window)
        prob = F.softmax(pred, dim=1)
        top_p, top_class = prob.topk(1, dim=1)
        top_class = decode(top_class)
        print(f'predicted: {top_class}, probability: {top_p[0][0]}')
        # print(pred)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    test_img_path = '../ogaorac-project1/'
    model_path = 'models/model4_softmax.pth'
    solver = Solver(model_path, test_img_path)
    solver.solve()

if __name__ == "__main__":
    main()

from model import NeuralNet
import torch
import numpy as np
import os
import cv2


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
        predictions = []
        img = imgs[4]
        width = 30
        window = np.zeros((1, 1, 64, width))
        window[0][0][:][:] = img[:, :width]
        window.resize((1, 1, 64, 26))
        window = window.astype(np.double)
        window = torch.tensor(window, dtype=torch.float)

        pred = model(window)
        pred = torch.exp(pred)
        print(pred)




def main():
    test_img_path = '../ogaorac-project1/'
    model_path = 'models/model2.pth'
    solver = Solver(model_path, test_img_path)
    solver.solve()

if __name__ == "__main__":
    main()

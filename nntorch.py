import torch
import torch.nn as nn
import torch.nn.functional as F

class NN(nn.Module):
    def __init__(self, output_shape):
        super(NN, self).__init__()
        self.output_shape = output_shape.n

        self.layers = nn.Sequential(
            nn.Conv2d(1,16,3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128,256,3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256,512,3), nn.ReLU(), nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(27648, 100), nn.ReLU(),
            nn.Linear(100, self.output_shape), nn.ReLU()
        )


    def forward(self, x):
        x = torch.Tensor(x)
        y = self.layers(x)
        return y

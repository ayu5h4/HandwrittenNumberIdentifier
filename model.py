import torch
import torch.nn as nn

class MNIST_model_54(nn.Module):
    def __init__(self):
        super(MNIST_model_54, self).__init__()
        self.stack_1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(10, 10, kernel_size=(3, 2), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        )
        self.stack_2 = nn.Sequential(
            nn.Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        )
        self.classification = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=7020, out_features=10, bias=True)
        )

    def forward(self, x):
        x = self.stack_1(x)
        x = self.stack_2(x)
        x = self.classification(x)
        return x

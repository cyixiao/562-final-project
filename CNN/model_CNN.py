import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Dropout(0.2),

            nn.Conv2d(32, 32, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),


            nn.Conv2d(32, 64, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Dropout(0.3),

            nn.Conv2d(64, 64, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3),


            nn.Conv2d(64, 128, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Dropout(0.4),

            nn.Conv2d(128, 128, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    model = CNN()
    test_input = torch.ones((64, 3, 32, 32))
    test_output = model(test_input)
    print(test_output.shape)

import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self, in_channels, num_classes) -> None:
        super().__init__()
        self.tanh = nn.Tanh()
        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6,
                               kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16,
                               kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120,
                               kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.linear1 = nn.Linear(120, 84)
        self.linear2 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.tanh(self.conv1(x))
        x = self.pool(x)
        x = self.tanh(self.conv2(x))
        x = self.pool(x)
        x = self.tanh(self.conv3(x))
        x = x.reshape(x.shape[0], -1)
        x = self.tanh(self.linear1(x))
        x = self.linear2(x)
        return x


if __name__ == "__main__":
    x = torch.randn(64, 1, 32, 32)
    model = LeNet()
    print(model(x).shape)

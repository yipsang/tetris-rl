from torch import nn
import torch.nn.functional as F


class ValueNetwork(nn.Module):
    def __init__(self, input_size, layers_size=[128, 256]):
        super().__init__()
        layers = []
        layers_size_ = [input_size] + layers_size
        for i in range(len(layers_size)):
            layers.append(nn.Linear(layers_size_[i], layers_size_[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layers_size_[-1], 1))
        self.val_net = nn.Sequential(*layers)

    def forward(self, x):
        """
        x PytorchTensor(*input_shape x channel): Channel last 2d state
        """
        return self.val_net(x)

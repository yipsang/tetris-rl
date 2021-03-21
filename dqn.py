from torch import nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, action_size, input_shape=(20, 10), in_channels=3):
        super().__init__()
        self.input_shape = input_shape
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=1)
        # self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=1, stride=1)
        # self.bn3 = nn.BatchNorm2d(64)

        output_shape = input_shape
        for kernel_size, stride, padding in [(5, 1, 0), (3, 1, 1), (1, 1, 0)]:
            output_shape = self._calculate_cnn_output_shape(
                output_shape, kernel_size, stride, padding=padding
            )

        self.fc4 = nn.Linear(output_shape[0] * output_shape[1] * 64, 256)
        self.head = nn.Linear(256, action_size)

    def _calculate_cnn_output_size(self, input_size, kernel_size, stride, padding=0):
        return int((input_size - kernel_size + 2 * padding) / stride + 1)

    def _calculate_cnn_output_shape(self, input_shape, kernel_size, stride, padding=0):
        height, width = input_shape
        return (
            self._calculate_cnn_output_size(
                height, kernel_size, stride, padding=padding
            ),
            self._calculate_cnn_output_size(
                width, kernel_size, stride, padding=padding
            ),
        )

    def forward(self, x):
        """
        x PytorchTensor(*input_shape x channel): Channel last 2d state
        """
        x = F.relu(self.conv1(x.permute((0, 3, 1, 2))))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.reshape(x.size(0), -1)))
        return self.head(x)

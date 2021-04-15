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

        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        x PytorchTensor(batch x input_shape): batch of input vectors
        """
        return self.val_net(x)


class ConvValueNetwork(nn.Module):
    def __init__(
        self,
        input_shape=(20, 10),
        in_channels=1,
        layers_config=[(2, 1, 0, 32), (2, 1, 1, 64), (1, 1, 0, 64)],
        linear_layer_size=128,
    ):
        """
        layers_config (list of tuples): [(kernel_size, stride, padding, layer_channels)]
        """
        super().__init__()
        layers = []
        for i in range(len(layers_config)):
            k, s, p, c = layers_config[i]
            if i == 0:
                in_c = in_channels
            else:
                in_c = layers_config[i - 1][3]
            layers.append(nn.Conv2d(in_c, c, kernel_size=k, stride=s, padding=p))
            layers.append(nn.ReLU())

        output_shape = input_shape
        for kernel_size, stride, padding, _ in layers_config:
            output_shape = self._calculate_cnn_output_shape(
                output_shape, kernel_size, stride, padding=padding
            )
        layers.append(nn.Flatten())
        layers.append(
            nn.Linear(
                output_shape[0] * output_shape[1] * layers_config[-1][3],
                linear_layer_size,
            )
        )
        layers.append(nn.ReLU())
        layers.append(nn.Linear(linear_layer_size, 1))
        self.val_net = nn.Sequential(*layers)

        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

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
        return self.val_net(x.permute((0, 3, 1, 2)))

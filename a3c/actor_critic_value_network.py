from torch import nn
import torch.nn.functional as F


class ActorCriticValueNetwork(nn.Module):
    def __init__(self, input_size, num_actions, layers_size=[128, 256]):
        super().__init__()
        layers = []
        layers_size_ = [input_size] + layers_size
        for i in range(len(layers_size)):
            layers.append(nn.Linear(layers_size_[i], layers_size_[i + 1]))
            layers.append(nn.ReLU())
        
        self.val_net = nn.Sequential(*layers)

        self.critic_linear = nn.Linear(layers_size_[-1], 1) # value function
        self.actor_linear = nn.Linear(layers_size_[-1], num_actions) # policy

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
        x = self.val_net(x.float())
        return self.critic_linear(x), self.actor_linear(x)

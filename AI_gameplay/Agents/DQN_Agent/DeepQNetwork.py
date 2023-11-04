import torch
from torch import nn

# Deep Q Network
class DeepQNetwork(torch.nn.Module):
    def __init__(self, lr, input_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions

        # Convolutional layers
        self.hidden_layer1 = nn.Sequential(
            # in_channels = 4 because we stack 4 frames together
            nn.Conv2d(in_channels= 4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU()
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU()
        )
        self.hidden_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        # Fully connected layers
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, n_actions)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.HuberLoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = self.hidden_layer1(state)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = torch.flatten(x, 1)  # Flatten
        x = torch.relu(self.fc1(x))
        actions = self.fc2(x)

        return actions

    def save_checkpoint(self,agent_type):
      torch.save(self.state_dict(), f'Model_for_Agents/{agent_type}_model/model')

    def load_checkpoint(self, agent_type):
      self.load_state_dict(torch.load(f'Model_for_Agents/{agent_type}_model/model'))
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class BaseNN(nn.Module):
    """Superclass for the Actor and Critic classes"""
    def __init__(self):
        super(BaseNN, self).__init__()
        self.config = Config()
        self.to(self.config.device)
        torch.manual_seed(self.config.seed)
        self.module_list = nn.ModuleList()

    def create_fc_layer(self, nodes_in, nodes_out):
        layer = nn.Linear(nodes_in, nodes_out)
        self.reset_parameters(layer)
        self.module_list.append(layer)

    def reset_parameters(self, layer):
        layer.weight.data.uniform_(-3e-3, 3e-3)


class Actor(BaseNN):
    """Build an actor (policy) network that maps states -> actions."""
    def __init__(self):
        super(Actor, self).__init__()
        for nodes_in, nodes_out in self.layers_nodes():
            self.create_fc_layer(nodes_in, nodes_out)

    def layers_nodes(self):
        nodes = []
        nodes.append(self.config.state_size)
        nodes.extend(self.config.actor_layers)
        nodes.append(self.config.action_size)
        nodes_in = nodes[:-1]
        nodes_out = nodes[1:]
        return zip(nodes_in, nodes_out)

    def forward(self, x):
        for layer in self.module_list[:-1]:
            x = F.relu(layer(x))
        x = self.module_list[-1](x)
        return torch.tanh(x)


class Critic(BaseNN):
    """Build a critic (value) network that maps
       (state, action) pair -> Q-values.
    """
    def __init__(self):
        super(Critic, self).__init__()
        for nodes_in, nodes_out in self.layers_nodes():
            self.create_fc_layer(nodes_in, nodes_out)

    def layers_nodes(self):
        nodes = []
        nodes.append(self.config.state_size)
        nodes.extend(self.config.critic_layers)
        nodes.append(1)
        nodes_in = nodes[:-1]
        nodes_in[1] += self.config.num_agents * self.config.action_size
        nodes_out = nodes[1:]
        return zip(nodes_in, nodes_out)

    def forward(self, state, action):
        x = F.relu(self.module_list[0](state))
        x = torch.cat((x, action), dim=1)
        for layer in self.module_list[1:-1]:
            x = F.relu(layer(x))
        x = self.module_list[-1](x)
        return x

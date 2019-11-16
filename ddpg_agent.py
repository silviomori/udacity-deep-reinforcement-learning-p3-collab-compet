import copy
import random

import torch
import torch.optim as optim

from config import Config
from model import Actor, Critic


class Agent():
    """Interacts with and learns from the environment."""
    def __init__(self):
        self.config = Config()
        random.seed(self.config.seed)

        # Actor Network
        self.actor_local = Actor()
        self.actor_target = Actor()
        local_state_dict = self.actor_local.state_dict()
        self.actor_target.load_state_dict(local_state_dict)
        self.actor_optimizer = optim.Adam(
            self.actor_local.parameters(),
            lr=self.config.actor_lr)

        # Critic Network
        self.critic_local = Critic()
        self.critic_target = Critic()
        local_state_dict = self.critic_local.state_dict()
        self.critic_target.load_state_dict(local_state_dict)
        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(),
            lr=self.config.critic_lr)

    def soft_update(self):
        """Soft update actor and critic parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        tau = self.config.tau
        for target_param, local_param \
                in zip(self.actor_target.parameters(),
                       self.actor_local.parameters()):
            target_param.data.copy_(
                tau * local_param.data
                + (1.0 - tau) * target_param.data)
        for target_param, local_param \
                in zip(self.critic_target.parameters(),
                       self.critic_local.parameters()):
            target_param.data.copy_(
                tau * local_param.data
                + (1.0 - tau) * target_param.data)

    def act(self, state):
        """Returns actions for given state as per current policy."""
        with torch.no_grad():
            self.actor_local.eval()
            state = torch.from_numpy(state).float()
            state.to(self.config.device)
            action = self.actor_local(state).data.cpu().numpy()
            self.actor_local.train()
        return action

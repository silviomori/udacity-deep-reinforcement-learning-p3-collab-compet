import random
from collections import namedtuple, deque

import torch

from config import Config


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self):
        self.config = Config()
        random.seed(self.config.seed)
        self.memory = deque(maxlen=self.config.buffer_size)

        self.experience = namedtuple(
            'Experience',
            field_names=['state', 'actions', 'rewards', 'next_state'])

    def store(self, state, actions, reward, next_state):
        """Add a new experience to memory."""
        e = self.experience(state, actions, reward, next_state)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, self.config.batch_size)

        states = self.create_tensor(dim=self.config.state_size)
        actions = self.create_tensor(dim=self.config.action_size)
        rewards = self.create_tensor()
        next_states = self.create_tensor(dim=self.config.state_size)
        for i, e in enumerate(experiences):
            states[i] = torch.as_tensor(e.state)
            actions[i] = torch.as_tensor(e.actions)
            rewards[i] = torch.as_tensor(e.rewards)
            next_states[i] = torch.as_tensor(e.next_state)
        return (states, actions, rewards, next_states)

    def create_tensor(self, dim=0):
        batch_size = self.config.batch_size
        num_agents = self.config.num_agents
        if dim > 0:
            size = (batch_size, num_agents, dim)
        else:
            size = (batch_size, num_agents)

        tensor = torch.empty(size=size, dtype=torch.float,
                             device=self.config.device,
                             requires_grad=False)
        return tensor

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

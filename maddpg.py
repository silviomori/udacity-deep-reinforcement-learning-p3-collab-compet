import copy
import random

import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from ddpg_agent import Agent


class MultiAgentDDPG():
    """Manage multi agents while interacting with the environment."""
    def __init__(self):
        super(MultiAgentDDPG, self).__init__()
        self.config = Config()
        self.agents = [Agent() for _ in range(self.config.num_agents)]

    def act(self, state):
        actions = [agent.act(obs) \
                   for agent, obs in zip(self.agents, state)]
        return actions

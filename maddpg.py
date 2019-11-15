from config import Config
from ddpg_agent import Agent
from replay_buffer import ReplayBuffer


class MultiAgentDDPG():
    """Manage multi agents while interacting with the environment."""
    def __init__(self):
        super(MultiAgentDDPG, self).__init__()
        self.config = Config()
        self.agents = [Agent() for _ in range(self.config.num_agents)]
        self.buffer = ReplayBuffer()

    def act(self, state):
        actions = [agent.act(obs) \
                   for agent, obs in zip(self.agents, state)]
        return actions

    def store(self, state, actions, rewards, next_state):
        self.buffer.store(state, actions, rewards, next_state)

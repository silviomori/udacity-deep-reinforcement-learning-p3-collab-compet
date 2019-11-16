import torch
import torch.nn.functional as F

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

    def actions_target(self, states):
        batch_size = self.config.batch_size
        num_agents = self.config.num_agents
        action_size = self.config.action_size
        with torch.no_grad():
            actions = torch.empty(
                (batch_size, num_agents, action_size),
                device=self.config.device)
            for idx, agent in enumerate(self.agents):
                actions[:,idx] = agent.actor_target(states[:,idx])
        return actions

    def actions_local(self, states, agent_id):
        batch_size = self.config.batch_size
        num_agents = self.config.num_agents
        action_size = self.config.action_size

        actions = torch.empty(
            (batch_size, num_agents, action_size),
            device=self.config.device)
        for idx, agent in enumerate(self.agents):
            action = agent.actor_target(states[:,idx])
            if not idx == agent_id:
                action.detach()
            actions[:,idx] = action

        return actions

    def store(self, state, actions, rewards, next_state):
        self.buffer.store(state, actions, rewards, next_state)

        if len(self.buffer) >= self.config.batch_size:
            self.learn()

    def learn(self):
        batch_size = self.config.batch_size
        for agent_id, agent in enumerate(self.agents):
            states, actions, rewards, next_states = self.buffer.sample()
            obs = states[:,agent_id] # observations for this agent
            r = rewards[:,agent_id].unsqueeze_(1)

            ## Train the Critic network
            with torch.no_grad():
                next_actions = self.actions_target(next_states)
                next_actions = next_actions.view(batch_size, -1)
                next_q_values = agent.critic_target(obs, next_actions)
                y = r + self.config.gamma * next_q_values
            actions = actions.view(batch_size, -1)
            q_value_predicted = agent.critic_local(obs, actions)

            agent.critic_optimizer.zero_grad()
            loss = F.mse_loss(q_value_predicted, y)
            loss.backward()
            agent.critic_optimizer.step()

            ## Train the Actor network
            actions_local = self.actions_local(states, agent_id)
            actions_local = actions_local.view(batch_size, -1)
            q_value_predicted = agent.critic_local(obs, actions)

            agent.actor_optimizer.zero_grad()
            loss = -q_value_predicted.mean()
            loss.backward()
            agent.actor_optimizer.step()

            agent.soft_update()

    def reset_noise(self):
        for agent in self.agents:
            agent.reset_noise()

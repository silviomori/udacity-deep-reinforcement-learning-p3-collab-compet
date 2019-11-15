from collections import namedtuple, deque

from config import Config


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self):
        self.config = Config()
        self.memory = deque(maxlen=self.config.buffer_size)

        self.experience = namedtuple(
            'Experience',
            field_names=['state', 'actions', 'rewards', 'next_state'])

    def store(self, state, actions, reward, next_state):
        """Add a new experience to memory."""
        e = self.experience(state, actions, reward, next_state)
        self.memory.append(e)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

import random
from collections import namedtuple, deque

Transition = namedtuple("Transition", ["state", "action", "next_state", "reward"])


class ReplayMemory:
    def __init__(self, capacity: int, generator: random.Random = None):
        if generator is None:
            generator = random.Random()

        self._generator = generator
        self.memory = deque([], maxlen=capacity)

    def push(self, state: str, action: int, next_state: str, reward: float):
        """Save a transition
        """
        self.memory.append(Transition(state, action, next_state, reward))

    def sample(self, batch_size):
        return self._generator.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

import random
import numpy as np

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def add(self, s0, a, r, s1, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        # self.buffer.append((s0[None, :], a, r, s1[None, :], done))
        self.buffer.append((s0, a, r, s1, done))

    def sample(self, batch_size):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        s0, a, r, s1, done = zip(*random.sample(self.buffer, batch_size))
        # arr1 = np.array(s0)
        return s0, a, r, s1, done

    def size(self):
        return len(self.buffer)
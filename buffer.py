import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.items = []
        self.capacity = capacity

    def add(self, item):
        self.items.append(item)
        if len(self.items) > self.capacity:
            self.items.pop(0)

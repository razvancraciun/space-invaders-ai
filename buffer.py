import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.items = []
        self.capacity = capacity
        self.count = 0

    def add(self, item):
        if self.count < self.capacity:
            self.items.append(item)
        else:
            self.items[self.count % self.capacity] = item
        self.count += 1
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.items = None
        self.capacity = capacity

    def add(self, item):
        if self.items == None:
            self.items = np.array([item])
        else:
            self.items = np.concatenate( (self.items, [item] ))
        if len(self.items) > self.capacity:
            self.items.pop(0)

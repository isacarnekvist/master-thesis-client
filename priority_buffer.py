import numpy as np


class PriorityNode():
    
    def __init__(self, value=None, left=None, right=None):
        self.left = left
        self.right = right
        if left and right:
            self.left.parent = self
            self.right.parent = self
            self._value = left._value + right._value
            self._max_value = max(self.left._max_value, self.right._max_value)
        elif left is None and right is None: # leaf
            if value is None:
                raise ValueError('value is required if left and right is None')
            self._value = value
            self._max_value = value
        else:
            raise ValueError('required: left, right | value')
        self.parent = None
        
    def sample(self, inverse=False):
        if self.left and self.right:
            if self._value == 0.0:
                raise ValueError('Cannot sample when children have value 0.0')
            rnd = np.random.rand()
            if rnd * self._value <= self.left._value:
                if inverse and self.right._value > 0.0:
                    return self.right.sample(inverse=inverse)
                else:
                    return self.left.sample(inverse=inverse)
            else:
                if inverse:
                    return self.left.sample(inverse=inverse)
                else:
                    return self.right.sample(inverse=inverse)
        else:
            return self
        
    def set_data(self, data):
        self.data = data
        
    def set_value(self, value):
        self._value = value
        self._max_value = value
        self.parent._recalculate_values()
        
    @property
    def value(self):
        return self._value
        
    def _recalculate_values(self):
        self._value = self.left._value + self.right._value
        self._max_value = max(self.left._max_value, self.right._max_value)
        if self.parent:
            self.parent._recalculate_values()
        
    def __repr__(self):
        return '<PriorityNode instance with value: {}>'.format(self._value)


class PriorityBuffer():
    
    def __init__(self, max_size):
        self.size = 0
        self.max_size = max_size
        if np.log2(max_size) != int(np.log2(max_size)):
            raise ValueError('max_size must be 2^k')
        self.leaves = [PriorityNode(0.0) for i in range(max_size)]
        previous_level = self.leaves
        for i in range(0, int(np.log2(max_size))):
            new_level = []
            for j in range(0, int(max_size / 2 ** i), 2):
                new_level.append(PriorityNode(
                    left=previous_level[j],
                    right=previous_level[j + 1]
                ))
            previous_level = new_level
        self.root = previous_level[0]
        
    def sample(self, inverse=False):
        """
        inverse : Bool
            if False, sample with probabilities proportional to values
            if True, sample inversely proportional to values
        """
        if self.size == 0:
            raise ValueError('Cannot sample from empty PriorityBuffer')
        return self.root.sample(inverse=inverse)
    
    def max_value(self):
        return self.root._max_value

    def sum(self):
        return self.root.value
        
    def add(self, data):
        """
        Returns the node set with the new data
        """
        if self.size < self.max_size:
            idx = self.size
            self.leaves[self.size].data = data
            self.size += 1
            return self.leaves[idx]
        else:
            old_node = self.sample(inverse=True)
            old_node.data = data
            return old_node

    def __repr__(self):
        return '<PriorityBuffer object with {} objects, max_value: {}>'.format(self.size, self.max_value())

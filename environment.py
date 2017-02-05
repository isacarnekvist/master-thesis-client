import numpy as np


def random_in_range(a, b):
    if b < a:
        raise ValueError('b should not be less than a')
    return np.random.rand() * (b - a) + a


class Environment():
     
    def __init__(self, goal=None):
        self._x_min = -0.15
        self._x_max = 0.15
        self._y_min = 0.15
        self._y_max = 0.30
        if goal:
            self.goal_x, self.goal_y = goal
            self.random_goal = False
        else:
            self.random_goal = True
        self.reset()
        
    def reset(self):
        if self.random_goal:
            self.goal_x = random_in_range(self._x_min, self._x_max)
            self.goal_y = random_in_range(self._y_min, self._y_max)
        self.eef_x = random_in_range(self._x_min, self._x_max)
        self.eef_y = random_in_range(self._y_min, self._y_max)
        
    def move(self, x, y):
        x = np.sign(x) * min(0.08, abs(x))
        y = np.sign(y) * min(0.08, abs(y))
        d = np.sqrt(x ** 2 + y ** 2)
        target_distance1 = np.sqrt(
            (self.eef_x - self.goal_x) ** 2 +
            (self.eef_y - self.goal_y) ** 2
        )
        if d > 0.05:
            x = 0.05 * x / d
            y = 0.05 * y / d
        self.eef_x += x
        self.eef_y += y
        outside_reward = -2
        if not (self._x_min <= self.eef_x <= self._x_max):
            return outside_reward
        if not (self._y_min <= self.eef_y <= self._y_max):
            return outside_reward
        target_distance2 = (
            (self.eef_x - self.goal_x) ** 2 +
            (self.eef_y - self.goal_y) ** 2
        )
        return np.exp(-1000 * target_distance2) - 1
    
    def plot(self):
        import matplotlib.pyplot as plt
        plt.axis('equal')
        plt.plot(
            [self._x_min, self._x_max, self._x_max, self._x_min, self._x_min],
            [self._y_min, self._y_min, self._y_max, self._y_max, self._y_min],
        )
        plt.plot(self.goal_x, self.goal_y, 'go')
        plt.plot(self.eef_x, self.eef_y, 'k+')

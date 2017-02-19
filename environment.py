import numpy as np

WIN = 0
LOSE = 1
NEUTRAL = 2
MAX_DIST = 0.01


def create_state_vector(eef_x, eef_y, circle_x, circle_y, goal_x, goal_y):
    return np.array([
        [eef_x, eef_y, circle_x, circle_y, goal_x, goal_y]
    ], dtype=np.float32)


class Circle:
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = 0.02
        
    def interact(self, x, y):
        theta = np.arctan2(y - self.y, x - self.x)
        center_distance = np.linalg.norm([self.y - y, self.x - x])
        distance = self.radius - center_distance
        if center_distance > self.radius:
            return
        self.x -= distance * np.cos(theta)
        self.y -= distance * np.sin(theta)
        
class Environment:
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        # Random on inner and outer circle
        eef_theta = np.random.rand() * 2 * np.pi
        self.eef_x = 0.10 * np.cos(eef_theta)
        self.eef_y = 0.20 + 0.07 * np.sin(eef_theta)
        circle_theta = np.random.rand() * 2 * np.pi
        circle_x = 0.04 * np.cos(circle_theta)
        circle_y = 0.20 + 0.02 * np.sin(circle_theta)
        self.circle = Circle(circle_x, circle_y)
        while True:
            goal_theta = np.random.rand() * 2 * np.pi
            self.goal_x = 0.04 * np.cos(goal_theta)
            self.goal_y = 0.20 + 0.02 * np.sin(goal_theta)
            if np.linalg.norm([self.goal_x - circle_x, self.goal_y - circle_y]) > 0.04:
                break
        while True:
            self.eef_x  = -0.10 + np.random.rand() * 0.20
            self.eef_y  =  0.12 + np.random.rand() * 0.17
            if np.linalg.norm([self.eef_x - circle_x, self.eef_y - circle_y]) < 0.04:
                continue
            else:
                break

    def get_state(self):
        return create_state_vector(
            self.eef_x,
            self.eef_y,
            self.circle.x,
            self.circle.y,
            self.goal_x,
            self.goal_y,
        )

    def interact(self, dx, dy):
        dist = np.linalg.norm([dx, dy])
        if dist > MAX_DIST:
            dx = MAX_DIST * dx / dist
            dy = MAX_DIST * dy / dist
        self.eef_x += dx
        self.eef_y += dy
        self.circle.interact(self.eef_x, self.eef_y)
        state = NEUTRAL
        reward = -4
        if not -0.15 <= self.eef_x <= 0.15:
            state = LOSE
        elif not 0.10 <= self.eef_y <= 0.30:
            state = LOSE
        elif not -0.15 <= self.circle.x <= 0.15:
            state = LOSE
        elif not 0.10 <= self.circle.y <= 0.30:
            state = LOSE
        elif np.linalg.norm([self.goal_x - self.circle.x, self.goal_y - self.circle.y]) < 0.005:
            state = WIN
            
        if state != LOSE:
            eef2circle = np.linalg.norm([self.eef_x - self.circle.x, self.eef_y - self.circle.y])
            circle2goal = np.linalg.norm([self.goal_x - self.circle.x, self.goal_y - self.circle.y])
            reward = (
                np.exp(-200 * eef2circle ** 2) - 1 +
                2 * np.exp(-200 * circle2goal ** 2) - 1
            )
        
        return state, reward, self.get_state()
        
    def plot(self, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots()
        plt.grid()
        ax.add_artist(plt.Circle(
            (self.goal_x, self.goal_y),
            self.circle.radius,
            color='k',
        ))
        ax.add_artist(plt.Circle(
            (self.goal_x, self.goal_y),
            self.circle.radius - 0.001,
            color='w',
        ))
        ax.add_artist(plt.Circle(
            (self.circle.x, self.circle.y),
            self.circle.radius,
            color='r',
            alpha=0.5
        ))
        plt.plot(self.eef_x, self.eef_y, 'k+', markersize=10)
        plt.xlim((-0.15, 0.15))
        plt.ylim((0.10, 0.30))

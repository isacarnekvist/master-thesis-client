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

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = np.round(value, 2)

    @property
    def y(self):
        return self._y
        
    @y.setter
    def y(self, value):
        self._y = np.round(value, 2)

        
class Environment:
    
    def __init__(self, max_dist):
        self.max_dist = max_dist
        self.reset()
    
    def reset(self):
        # Random on inner and outer circle
        circle_x = 0.20 * np.random.rand() - 0.10
        circle_y = 0.10 * np.random.rand() + 0.16
        self.goal_x = 0.00
        self.goal_y = 0.22
        self.circle = Circle(circle_x, circle_y)
        eef_theta = np.random.rand() * 2 * np.pi
        self.eef_x = circle_x
        self.eef_y = circle_y
        while np.linalg.norm([self.eef_x - circle_x, self.eef_y - circle_y]) < 0.03:
            self.eef_x = 0.20 * np.random.rand() - 0.10
            self.eef_y = 0.10 * np.random.rand() + 0.16

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
        if dist > self.max_dist:
            dx = self.max_dist * dx / dist
            dy = self.max_dist * dy / dist
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
                2 * (np.exp(-200 * circle2goal ** 2) - 1)
            )
        
        return state, reward, self.get_state()
    
    def heuristic_move(self):
        e = self
        a = np.array([e.eef_x, e.eef_y])
        b = np.array([e.circle.x, e.circle.y])
        d = b - a
        d_norm = np.linalg.norm(d)
        theta = np.arcsin(0.02 / d_norm)
        A = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        xa = np.dot(A, d) * np.sqrt(d_norm ** 2 - 0.02 ** 2) / d_norm
        xb = np.dot(A.T, d) * np.sqrt(d_norm ** 2 - 0.02 ** 2) / d_norm

        fg = np.array([e.goal_x, e.goal_y])
        if np.linalg.norm(fg - b) < 0.0005:
            return np.zeros(2)
        pd = (fg - b) / np.linalg.norm(fg - b) # pushing direction
        pg = b - pd * 0.02                     # pushing goal
        e_dist = np.linalg.norm(xa)
        a_dist = np.linalg.norm(pg - a - xa)
        b_dist = np.linalg.norm(pg - a - xb)
        pg_dist = np.linalg.norm(a - pg)
        if pg_dist < 0.005:
            return min(self.max_dist, d_norm) * pd
        if 0.002 < e_dist < pg_dist:
            if a_dist < b_dist:
                return min(self.max_dist, np.linalg.norm(xa)) * xa / np.linalg.norm(xa)
            else:
                return min(self.max_dist, np.linalg.norm(xb)) * xb / np.linalg.norm(xb)
        else:
            return min(self.max_dist, np.linalg.norm(pg - a)) * (pg - a) / np.linalg.norm(pg - a)


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

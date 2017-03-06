import numpy as np

WIN = 0
LOSE = 1
NEUTRAL = 2
MAX_DIST = 0.01


def random(a, b):
    if b < a:
        raise ValueError('b must be <= a')
    return a + (b - a) * np.random.rand()


class Circle:
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = 0.03
        
    def interact(self, x, y):
        theta = np.arctan2(self.y - y, self.x - x)
        center_distance = np.linalg.norm([self.y - y, self.x - x])
        distance = self.radius - center_distance
        if center_distance > self.radius:
            return
        self.x = self.x + distance * np.cos(theta)
        self.y = self.y + distance * np.sin(theta)

        
class Environment:
    
    def __init__(self, max_dist, mode):
        self.mode = mode.lower()
        if self.mode == 'reaching-fixed-goal':
            pass
        elif self.mode == 'reaching-moving-goal':
            pass
        elif self.mode == 'pushing-fixed-goal':
            pass
        elif self.mode == 'pushing-fixed-cube':
            pass
        elif self.mode == 'pushing-moving-goal':
            raise NotImplementedError
        else:
            raise ValueError('Not a valid mode string')
        self.min_x, self.max_x = (-0.15, 0.15)
        self.min_y, self.max_y = (0.10, 0.30)
        self.max_dist = max_dist
        self.reset()
    
    def reset(self):
        self.circle = Circle(0.0, 0.2)
        if self.mode == 'reaching-fixed-goal':
            self.reset_reaching_fixed_goal()
        if self.mode == 'reaching-moving-goal':
            self.reset_reaching_moving_goal()
        elif self.mode == 'pushing-fixed-goal':
            self.reset_pushing_fixed_goal()
        elif self.mode == 'pushing-fixed-cube':
            self.reset_pushing_fixed_cube()

    def reset_reaching_fixed_goal(self):
        self.goal_x = 0.00
        self.goal_y = 0.20
        self.eef_x = random(self.min_x, self.max_x)
        self.eef_y = random(self.min_y, self.max_y)

    def reset_reaching_moving_goal(self):
        self.goal_x = random(self.min_x, self.max_x)
        self.goal_y = random(self.min_y, self.max_y)
        self.eef_x = random(self.min_x, self.max_x)
        self.eef_y = random(self.min_y, self.max_y)

    def reset_pushing_fixed_cube(self):
        self.eef_x = float(-0.08 + np.random.randn() * 0.01)
        self.eef_y = float(0.20 + np.random.randn() * 0.02)
        circle_x = float(0.00 + np.random.randn() * 0.01)
        circle_y = float(0.20 + np.random.randn() * 0.01)
        self.circle = Circle(circle_x, circle_y)
        self.goal_x = 0.06
        self.goal_y = 0.20

    def reset_pushing_fixed_goal(self):
        self.goal_x = 0.00
        self.goal_y = 0.20
        self.eef_x = random(self.min_x, self.max_x)
        self.eef_y = random(self.min_y, self.max_y)
        circle_x = random(self.min_x + self.circle.radius, self.max_x - self.circle.radius)
        circle_y = random(self.min_y + self.circle.radius, self.max_y - self.circle.radius)
        self.circle = Circle(circle_x, circle_y)
        while np.linalg.norm([self.eef_x - self.circle.x, self.eef_y - self.circle.y]) < self.circle.radius:
            self.circle.x = random(self.min_x, self.max_x)
            self.circle.y = random(self.min_y, self.max_y)

    def get_state(self):
        if self.mode == 'reaching-fixed-goal':
            return np.array([[
                self.eef_x,
                self.eef_y,
            ]])
        if self.mode == 'reaching-moving-goal':
            return np.array([[
                self.eef_x,
                self.eef_y,
                self.goal_x,
                self.goal_y,
            ]])
        elif self.mode == 'pushing-fixed-goal':
            return np.array([[
                self.eef_x,
                self.eef_y,
                self.circle.x,
                self.circle.y,
            ]])
        elif self.mode == 'pushing-fixed-cube':
            return np.array([[
                self.eef_x,
                self.eef_y,
                self.circle.x,
                self.circle.y,
            ]])

    def interact(self, dx, dy):
        dist = np.linalg.norm([dx, dy])
        if dist > self.max_dist:
            dx = self.max_dist * dx / dist
            dy = self.max_dist * dy / dist
        self.eef_x += dx
        self.eef_y += dy
        if self.mode.startswith('pushing'):
            self.circle.interact(self.eef_x, self.eef_y)

        state = NEUTRAL
        reward = -2
        if not self.min_x <= self.eef_x <= self.max_x:
            state = LOSE
        elif not self.min_y <= self.eef_y <= self.max_y:
            state = LOSE
        elif not self.min_x <= self.circle.x <= self.max_x:
            state = LOSE
        elif not self.min_y <= self.circle.y <= self.max_y:
            state = LOSE
        elif self.mode.startswith('pushing') and np.linalg.norm([self.goal_x - self.circle.x, self.goal_y - self.circle.y]) < 0.005:
            state = WIN
        elif self.mode.startswith('reaching') and np.linalg.norm([self.goal_x - self.eef_x, self.goal_y - self.eef_x]) < 0.005:
            state = WIN
            
        if state != LOSE:
            eef2circle = np.linalg.norm([self.eef_x - self.circle.x, self.eef_y - self.circle.y])
            circle2goal = np.linalg.norm([self.goal_x - self.circle.x, self.goal_y - self.circle.y])
            eef2goal = np.linalg.norm([self.goal_x - self.eef_x, self.goal_y - self.eef_y])
            if self.mode.startswith('pushing'):
                reward = (
                    (np.exp(-200 * eef2circle ** 2) - 1) +
                    (np.exp(-200 * circle2goal ** 2) - 1)
                )
            else:
                reward = (
                    np.exp(-200 * eef2goal ** 2) - 1
                )
        
        return state, reward, self.get_state()
    
    def heuristic_move(self):
        e = self
        a = np.array([e.eef_x, e.eef_y])
        b = np.array([e.circle.x, e.circle.y])
        d = b - a
        d_norm = np.linalg.norm(d)
        theta = np.arcsin(self.circle.radius / d_norm)
        A = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        xa = np.dot(A, d) * np.sqrt(d_norm ** 2 - self.circle.radius ** 2) / d_norm
        xb = np.dot(A.T, d) * np.sqrt(d_norm ** 2 - self.circle.radius ** 2) / d_norm

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
            fig, ax = plt.subplots()
        ax.grid()
        if self.mode.startswith('pushing'):
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
        else:
            ax.add_artist(plt.Circle(
                (self.goal_x, self.goal_y),
                0.005,
                color='k',
            ))
            ax.add_artist(plt.Circle(
                (self.goal_x, self.goal_y),
                0.004,
                color='w',
            ))
        ax.plot(self.eef_x, self.eef_y, 'k+', markersize=10)
        ax.set_xlim((self.min_x, self.max_x))
        ax.set_ylim((self.min_y, self.max_y))

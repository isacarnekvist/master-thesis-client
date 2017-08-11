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
            pass
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
        elif self.mode == 'pushing-moving-goal':
            self.reset_pushing_moving_goal()
        return self.get_state()

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
        circle_x = 0.00 + np.random.randn() * 0.01
        circle_y = 0.20 + np.random.randn() * 0.01
        self.circle = Circle(circle_x, circle_y)
        theta = 2 * np.pi * np.random.rand()
        self.goal_x = 0.06
        self.goal_y = 0.20
        self.eef_x, self.eef_y = circle_x, circle_y
        while np.linalg.norm([self.eef_x - self.circle.x, self.eef_y - self.circle.y]) < self.circle.radius:
            self.eef_x = random(self.min_x, self.max_x)
            self.eef_y = random(self.min_y, self.max_y)

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
            
    def reset_pushing_moving_goal(self):
        self.goal_x = random(self.min_x + self.circle.radius, self.max_x - self.circle.radius)
        self.goal_y = random(self.min_y + self.circle.radius, self.max_y - self.circle.radius)
        self.eef_x = random(self.min_x, self.max_x)
        self.eef_y = random(self.min_y, self.max_y)
        circle_x = random(self.min_x + 2 * self.circle.radius, self.max_x - 2 * self.circle.radius)
        circle_y = random(self.min_y + 2 * self.circle.radius, self.max_y - 2 * self.circle.radius)
        self.circle = Circle(circle_x, circle_y)
        while (np.linalg.norm([self.eef_x - self.circle.x, self.eef_y - self.circle.y]) < self.circle.radius and
               np.linalg.norm([self.circle.x - self.goal_x, self.circle.y - self.goal_y]) > 0.01):
            self.circle.x = random(self.min_x + 2 * self.circle.radius, self.max_x - 2 *  self.circle.radius)
            self.circle.y = random(self.min_y + 2 * self.circle.radius, self.max_y - 2 * self.circle.radius)

    def get_state(self):
        if self.mode == 'reaching-fixed-goal':
            return np.array([
                self.eef_x,
                self.eef_y,
            ])
        if self.mode == 'reaching-moving-goal':
            return np.array([
                self.eef_x,
                self.eef_y,
                self.goal_x,
                self.goal_y,
            ])
        elif self.mode == 'pushing-fixed-goal':
            return np.array([
                self.eef_x,
                self.eef_y,
                self.circle.x,
                self.circle.y,
            ])
        elif self.mode == 'pushing-fixed-cube':
            return np.array([
                self.eef_x,
                self.eef_y,
                self.circle.x,
                self.circle.y,
            ])
        elif self.mode == 'pushing-moving-goal':
            return np.array([
                self.eef_x,
                self.eef_y,
                self.circle.x,
                self.circle.y,
                self.goal_x,
                self.goal_y,
            ])

    def step(self, a):
        dx, dy = a
        dist = np.linalg.norm([dx, dy])
        if dist > self.max_dist:
            dx = self.max_dist * dx / dist
            dy = self.max_dist * dy / dist
        self.eef_x += dx
        self.eef_y += dy
        circle_start = np.array([self.circle.x, self.circle.y])
        goal = np.array([self.goal_x, self.goal_y])
        if self.mode.startswith('pushing'):
            self.circle.interact(self.eef_x, self.eef_y)
        circle_end = np.array([self.circle.x, self.circle.y])

        state = NEUTRAL
        reward = -1
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
                    np.linalg.norm(circle_start - goal) - np.linalg.norm(circle_end - goal)
                ) / 0.01
            else:
                reward = (
                    np.exp(-200 * eef2goal ** 2) - 1
                )
        
        return self.get_state(), reward, state in [LOSE, WIN], state
    
    def plot(self, ax=None, eef_color='b'):
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
        ax.plot(self.eef_x, self.eef_y, '+', color=eef_color, markersize=10)
        ax.set_xlim((self.min_x, self.max_x))
        ax.set_ylim((self.min_y, self.max_y))

    def heuristic_move(self):
        def cap_sphere(v):
            if np.linalg.norm(v) > self.max_dist:
                return self.max_dist * v / np.linalg.norm(v)
            else:
                return v
        #
        #         h __
        #         /    \
        #       p(r c  )         a_norm^2 = d_norm^2 + r^2
        #        \    /
        #      d a--- p'
        #
        #    e          g
        #  alpha
        #
        #
        eps = 0.0005
        e = np.array([self.eef_x, self.eef_y])
        c = np.array([self.circle.x, self.circle.y])
        g = np.array([self.goal_x, self.goal_y])
        a = c - e
        r_norm = self.circle.radius
        h = c + r_norm * (c - g) / np.linalg.norm(c - g)
        a_norm = np.linalg.norm(a)
        d_norm = np.sqrt(max(0.0, a_norm ** 2 - r_norm ** 2))
        alpha = np.arctan2(r_norm, d_norm)
        rot_pos = np.array([[np.cos(alpha), -np.sin(alpha)], [ np.sin(alpha), np.cos(alpha)]])
        rot_neg = np.array([[np.cos(alpha),  np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]])
        pa = e + np.dot(rot_pos, a)
        pb = e + np.dot(rot_neg, a)
        if np.linalg.norm(pa - h) < np.linalg.norm(pb - h):
            p = pa
        else:
            p = pb
        if np.linalg.norm(c - g) < eps: # cube at goal
            return np.zeros(2)
        if np.linalg.norm(h - e) < eps: # at pushing position
            return cap_sphere(min(np.linalg.norm(g - c), self.max_dist) * (g - c) / (np.linalg.norm(g - c) + eps))
        if np.linalg.norm(e - h) < np.linalg.norm(e - p): # pushing position is closer than edge intercept
            return cap_sphere(min(np.linalg.norm(h - e), self.max_dist) * (h - e) / (np.linalg.norm(h - e) + eps))
        else:
            return cap_sphere(
                min(np.linalg.norm(p - e), self.max_dist) * (p - e) / (np.linalg.norm(p - e) + eps)
            )

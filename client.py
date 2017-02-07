import logging
from time import sleep

import numpy as np
from scipy.spatial.distance import euclidean

from naf import NNet
from arm_wrapper import Arm
from pyuarm.protocol import SERVO_BOTTOM, SERVO_LEFT, SERVO_RIGHT


logger = logging.getLogger('Client')
logger.setLevel(logging.INFO) # INFO


def create_state_vector(eef_x, eef_y, eef_z, goal_x, goal_y):
    return np.array([[eef_x, eef_y, eef_z, goal_x, goal_y]])


def random_in_range(a, b):
    return a + (b - a) * np.random.rand()


def is_lose_pose(x, y, z):
    if not (-0.12 <= x <= 0.12):
        return True
    if not (0.12 <= y <= 0.27):
        return True
    if not (0.025 <= z <= 0.038):
        return True
    return False


def random_pose():
    return (
        random_in_range(-0.10, 0.10),
        random_in_range(0.15, 0.25),
        random_in_range(0.029, 0.034)
    )


def reward(x, y, z, goal_x, goal_y):
    eef = np.array([x, y])
    goal = np.array([goal_x, goal_y])

    height_reward = -5e4 * (z - 0.0315) ** 2
    np.sum((eef - goal) ** 2)
    d = euclidean(eef, goal)
    d_reward = np.exp(-1000 * d ** 2) - 1
    return height_reward + d_reward


class Client():

    def __init__(self):
        self.arm = Arm()
        self.nn = NNet(x_size=3 + 2, u_size=3)
        sleep(2.0)

    def start(self):
        self.control_loop()

    def random_start_pose(self):
        x, y, z = random_pose()
        self.arm.move_to(x, y, z + 0.03, velocity=0.5)
        sleep(2.0)
        self.arm.move_to(x, y, z)
        sleep(0.3)

    def next_move(self, noise_factor=1.0):
        eef_x, eef_y, eef_z = self.arm.get_position()

        # new controls plus noise
        u_alpha, u_beta, u_gamma = 6.0 * self.nn.mu.predict(create_state_vector(
            eef_x, eef_y, eef_z, self.goal_x, self.goal_y
        ))[0, :] + noise_factor * 2.5 * np.random.randn(3)

        if abs(u_alpha) > 6.0:
            u_alpha = 6.0 * np.sign(u_alpha)
        if abs(u_beta) > 6.0:
            u_beta = 6.0 * np.sign(u_beta)
        if abs(u_gamma) > 6.0:
            u_gamma = 6.0 * np.sign(u_gamma)

        return u_alpha, u_beta, u_gamma

    def control_loop(self):
        for i in range(4):
            logger.warning('Setting random start and goal poses')
            self.random_start_pose()
            self.goal_x, self.goal_y, _ = random_pose()
            for j in range(10):
                self.arm.set_angles_relative(*self.next_move())
                sleep(0.03)
                xp, yp, zp = self.arm.get_position()
                state_prime = create_state_vector(xp, yp, zp, self.goal_x, self.goal_y)
                if is_lose_pose(xp, yp, zp):
                    logger.warning('outside workspace, reward -2')
                    break
                logger.warning('reward: {}'.format(reward(xp, yp, zp, self.goal_x, self.goal_y)))
                sleep(0.03)

    def stop(self):
        self.arm.stop()
        self.arm.disconnect()
        

c = Client()
try:
    c.start()
except KeyboardInterrupt:
    c.stop()

c.stop()

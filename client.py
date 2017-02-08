import json
import logging
import requests
from time import sleep
from datetime import datetime, timedelta

import logcolor
import numpy as np
from scipy.spatial.distance import euclidean

from naf import NNet
from arm_wrapper import Arm
from pyuarm.protocol import SERVO_BOTTOM, SERVO_LEFT, SERVO_RIGHT


logcolor.basic_config(level=logging.WARNING)
logger = logging.getLogger('client')
logger.setLevel(logging.DEBUG)


def create_state_vector(eef_x, eef_y, eef_z, goal_x, goal_y):
    return np.array([[eef_x, eef_y, eef_z, goal_x, goal_y]])


def random_in_range(a, b):
    return a + (b - a) * np.random.rand()


def is_lose_pose(x, y, z):
    if not (-0.12 <= x <= 0.12):
        return True
    if not (0.15 <= y <= 0.27):
        return True
    if not (0.025 <= z <= 0.038):
        return True
    return False

def is_win_pose(x, y, z, gx, gy):
    if euclidean([x, y], [gx, gy]) < 0.01:
        return True
    else:
        return False


def random_pose():
    return (
        random_in_range(-0.12, 0.12),
        random_in_range(0.17, 0.25),
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
        self.session = requests.Session()
        self.max_angle_change = 3.8
        sleep(2.0)

    def update_weights(self):
        logger.debug('Fetching new parameters')
        try:
            params = json.loads(self.session.get('http://beorn:5000/get_params').text)
        except requests.exceptions.ConnectionError:
            logger.error('Server not online? Could not update weights.')
            return
        self.nn.q.set_weights([np.array(param) for param in params])

    def start(self):
        self.control_loop()

    def random_start_pose(self):
        x, y, z = random_pose()
        self.arm.move_to(x, y, z + 0.03, velocity=0.5)
        sleep(4.0)
        self.arm.move_to(x, y, z)
        sleep(4.0)

    def next_move(self, noise_factor=1.0):
        eef_x, eef_y, eef_z = self.arm.get_position()

        # new controls plus noise
        u_alpha, u_beta, u_gamma = self.max_angle_change * self.nn.mu.predict(create_state_vector(
            eef_x, eef_y, eef_z, self.goal_x, self.goal_y
        ))[0, :] + noise_factor * 1.0 * np.random.randn(3)

        if abs(u_alpha) > self.max_angle_change:
            u_alpha = self.max_angle_change * np.sign(u_alpha)
        if abs(u_beta) > self.max_angle_change:
            u_beta = self.max_angle_change * np.sign(u_beta)
        if abs(u_gamma) > self.max_angle_change:
            u_gamma = self.max_angle_change * np.sign(u_gamma)

        return u_alpha, u_beta, u_gamma

    def control_loop(self):
        experience = []
        self.update_weights()
        latest_param_update = datetime.now()
        while True:
            logger.debug('Setting random start and goal poses')
            self.random_start_pose()
            self.goal_x, self.goal_y, _ = random_pose()
            while True:
                x, y, z = self.arm.get_position()
                u = self.next_move()
                logger.debug('Sending angles: {}'.format(u))
                if self.arm._arm.is_connected():
                    self.arm.set_angles_relative(*self.next_move())
                else:
                    self.arm.disconnect()
                    self.arm = Arm()
                    break
                sleep(0.1)
                xp, yp, zp = self.arm.get_position()
                state_prime = create_state_vector(xp, yp, zp, self.goal_x, self.goal_y)
                r = reward(xp, yp, zp, self.goal_x, self.goal_y)
                if is_lose_pose(xp, yp, zp):
                    r = -4
                # TODO Add break if in 'winning' terminal state
                logger.debug('reward: {}'.format(r))
                experience.append({
                    'x': [x, y, z, self.goal_x, self.goal_y],
                    'xp': [xp, yp, zp, self.goal_x, self.goal_y],
                    'u': list(u),
                    'r': r,
                })
                if is_lose_pose(xp, yp, zp):
                    logger.info('Reached outside workspace')
                    break
                if is_win_pose(xp, yp, zp, self.goal_x, self.goal_y):
                    logger.info('Reached target pose!')
                    break
            if datetime.now() > latest_param_update + timedelta(seconds=15):
                self.update_weights()
                latest_param_update = datetime.now()
            if len(experience) > 20:
                self.send_experience(experience)
                experience = []

    def send_experience(self, experience):
        headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
        try:
            r = self.session.put('http://beorn:5000/put_experience', data=json.dumps(experience), headers=headers)
            logger.debug('Sent experience {}'.format(r))
        except requests.exceptions.ConnectionError:
            logger.error('Server not online? Could not send experience.')

    def stop(self):
        self.arm.stop()
        self.arm.disconnect()
        

c = Client()
try:
    c.start()
except KeyboardInterrupt:
    c.stop()

c.stop()

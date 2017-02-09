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


logcolor.basic_config(level=logging.ERROR)
logger = logging.getLogger('client')
logger.setLevel(logging.INFO)


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
        self.max_angle_move = 3.5
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
        for i in range(64):
            if i % 32:
                self.do_one_trial()
            else:
                trial = self.do_one_trial(noise_factor=0.0)
                try:
                    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
                    r = self.session.put('http://beorn:5000/put_test_trial', data=json.dumps(trial), headers=headers)
                    logger.debug('Sent test trial {}'.format(r))
                except requests.exceptions.ConnectionError:
                    logger.error('Server not online? Could not send test trial.')

    def random_start_pose(self):
        x, y, z = random_pose()
        self.arm.move_to(x, y, z + 0.03, velocity=0.5)
        sleep(4.0)
        self.arm.move_to(x, y, z)
        sleep(4.0)

    def next_move(self, noise_factor=1.0):
        eef_x, eef_y, eef_z = self.arm.get_position()

        # new controls plus noise
        u_alpha, u_beta, u_gamma = self.max_angle_move * self.nn.mu.predict(create_state_vector(
            eef_x, eef_y, eef_z, self.goal_x, self.goal_y
        ))[0, :] + noise_factor * 2.0 * np.random.randn(3)

        if abs(u_alpha) > self.max_angle_move:
            u_alpha = self.max_angle_move * np.sign(u_alpha)
        if abs(u_beta) > self.max_angle_move:
            u_beta = self.max_angle_move * np.sign(u_beta)
        if abs(u_gamma) > self.max_angle_move:
            u_gamma = self.max_angle_move * np.sign(u_gamma)

        return u_alpha, u_beta, u_gamma

    def do_one_trial(self, noise_factor=1.0, max_movements=100):
        experience = []
        self.update_weights()
        logger.debug('Setting random start and goal poses')
        self.goal_x, self.goal_y, _ = random_pose()
        logger.info('New goal at x: {}, y: {}'.format(self.goal_x, self.goal_y))
        self.random_start_pose()
        for i in range(max_movements):
            x, y, z = self.arm.get_position()
            u = self.next_move(noise_factor=noise_factor)
            logger.debug('Sending relative angle commands: {}'.format(u))
            if self.arm._arm.is_connected():
                self.arm.set_angles_relative(*u)
            else:
                self.arm.disconnect()
                self.arm.stop()
                exit(-1)
            sleep(0.1)
            xp, yp, zp = self.arm.get_position()
            state_prime = create_state_vector(xp, yp, zp, self.goal_x, self.goal_y)
            r = reward(xp, yp, zp, self.goal_x, self.goal_y)
            if is_lose_pose(xp, yp, zp):
                r = -4
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
        self.send_experience(experience)
        return experience

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

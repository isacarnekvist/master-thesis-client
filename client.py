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


def remove_command_threshold(dx, dy):
    #dx += 0.0058 * np.sign(dx)
    #dy += 0.0058 * np.sign(dy)
    theta = np.arctan2(dy, dx)
    dx += 0.0055 * np.cos(theta)
    dy += 0.0055 * np.sin(theta)
    return dx, dy


def create_state_vector(eef_x, eef_y, goal_x, goal_y):
    return np.array([[eef_x, eef_y, goal_x, goal_y]])


def random_in_range(a, b):
    return a + (b - a) * np.random.rand()


def is_lose_pose(x, y):
    if not (-0.12 <= x <= 0.12):
        return True
    if not (0.15 <= y <= 0.27):
        return True
    return False

def is_win_pose(x, y, gx, gy):
    if euclidean([x, y], [gx, gy]) < 0.01:
        return True
    else:
        return False


def random_pose():
    return (
        random_in_range(-0.12, 0.12),
        random_in_range(0.17, 0.25),
        0.030
    )


def reward(x, y, goal_x, goal_y):
    eef = np.array([x, y])
    goal = np.array([goal_x, goal_y])

    d = euclidean(eef, goal)
    d_reward = np.exp(-1000 * d ** 2) - 1
    return d_reward


class Client():

    def __init__(self):
        self.arm = Arm()
        self.nn = NNet(x_size=2 + 2, u_size=2)
        self.session = requests.Session()
        self.max_axis_move = 0.012
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
        for i in range(16):
            if i % 4:
                self.do_one_trial(noise_factor=1.0)
            else:
                trial = self.do_one_trial(noise_factor=1.0)
                if trial is None:
                    continue
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
        sleep(1.5)

    def next_move(self, eef_x, eef_y, noise_factor=1.0):
        # new controls plus noise
        u_dx, u_dy = self.max_axis_move * self.nn.mu.predict(create_state_vector(
            eef_x, eef_y, self.goal_x, self.goal_y
        ))[0, :] + noise_factor * 0.006 * np.random.randn(2)

        euclid = np.sqrt(u_dx ** 2 + u_dy ** 2)
        if abs(u_dx) > self.max_axis_move:
            u_dx = self.max_axis_move * np.sign(u_dx)
        if abs(u_dy) > self.max_axis_move:
            u_dy = self.max_axis_move * np.sign(u_dy)

        return u_dx, u_dy

    def do_one_trial(self, noise_factor=1.0, max_movements=32):
        experience = []
        self.update_weights()
        logger.debug('Setting random start and goal poses')
        self.goal_x, self.goal_y, _ = random_pose()
        logger.info('New goal at x: {}, y: {}'.format(self.goal_x, self.goal_y))
        self.random_start_pose()
        for i in range(max_movements):
            x, y, _ = self.arm.get_position()
            dx, dy = self.next_move(x, y, noise_factor=noise_factor)
            dx_fixed, dy_fixed = remove_command_threshold(dx, dy)
            if self.arm._arm.is_connected():
                self.arm.move_to(x + dx_fixed, y + dy_fixed, 0.03)
            else:
                self.arm.disconnect()
                self.arm.stop()
                exit(-1)
            sleep(0.2)
            xp, yp, _ = self.arm.get_position()
            error_euclid = euclidean([xp - x, yp - y], [dx, dy])
            if error_euclid > 0.01:
                logger.warning('Large command/measure error: {:.4f} m, aborting trial'.format(euclidean([xp - x, yp - y], [dx, dy])))
                return
            state_prime = create_state_vector(xp, yp, self.goal_x, self.goal_y)
            r = reward(xp, yp, self.goal_x, self.goal_y)
            if is_lose_pose(xp, yp):
                r = -4
            logger.debug('reward: {}'.format(r))
            experience.append({
                'x': [x, y, self.goal_x, self.goal_y],
                'xp': [xp, yp, self.goal_x, self.goal_y],
                'u': [dx, dy],
                'r': r,
            })
            if is_lose_pose(xp, yp):
                logger.info('Reached outside workspace')
                break
            if is_win_pose(xp, yp, self.goal_x, self.goal_y):
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

import os
import sys
import json
import logging
import requests
from time import sleep
from datetime import datetime, timedelta

import logcolor
import numpy as np
from scipy.spatial.distance import euclidean

from arm_wrapper import Arm
from ddpg import Actor, Critic
from rplidar_wrapper import LidarWrapper
from pyuarm.protocol import SERVO_BOTTOM, SERVO_LEFT, SERVO_RIGHT


logcolor.basic_config(level=logging.ERROR)
logger = logging.getLogger('client')
logger.setLevel(logging.INFO)


def suspicious_transition(X):
    ex1, ey1, d1 = X['x']
    ex2, ey2, d2 = X['xp']
    d = abs(d1 - d2)
    return d > 0.03


def remove_command_threshold(dx, dy, max_axis_move):
    #dx += 0.0058 * np.sign(dx)
    #dy += 0.0058 * np.sign(dy)
    theta = np.arctan2(dy, dx)
    dx += 0.01 * np.cos(theta)
    dy += 0.01 * np.sin(theta)
    return dx, dy


def create_state_vector(eef_x, eef_y, cube_x, cube_y, goal_x, goal_y):
    cube = np.array([cube_x, cube_y])
    goal = np.array([goal_x, goal_y])
    eef = np.array([eef_x, eef_y])
    d = cube - goal
    d_norm = np.linalg.norm(d)
    if d_norm > 0:
        alpha = np.arctan2(d[1], d[0])
        rot = np.array([[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]])
        rel = np.dot(rot, eef - cube)
        return np.array([
            rel[0], # eef_x
            rel[1],    # eef_y
            d_norm  # circle to goal distance
        ])
    else:
        # If circle is at goal, let eef be on x-axis
        return np.array([
            np.linalg.norm(eef - cube), # eef_x
            0.0, # eef_y
            0.0  # circle to goal distance
        ])


def random_in_range(a, b):
    return a + (b - a) * np.random.rand()


def is_lose_pose(x, y, cube_x, cube_y):
    if not (-0.15 <= x <= 0.15):
        return True
    if not (0.11 <= y <= 0.32):
        return True
    if not (-0.11 <= cube_x <= 0.11):
        return True
    if not (0.13 <= cube_y <= 0.26):
        return True
    return False

def is_win_pose(x, y, cube_x, cube_y, gx, gy):
    if euclidean([cube_x, cube_y], [gx, gy]) < 0.005:
        return True
    else:
        return False


def random_pose():
    return (
        random_in_range(-0.12, 0.12),
        random_in_range(0.17, 0.25),
        0.030
    )


def cube_pose_retry():
    return lw.cube_pose, lw.scans


def random_cube_start_goal():
    return (
        random_in_range(-0.06, 0.06),  # cube_x
        random_in_range(0.17, 0.23),   # cube_y
        random_in_range(-0.06, 0.06),  # goal_x
        random_in_range(0.17, 0.23),   # goal_y
    )


def reward(x, y, cube_x, cube_y, cube_xp, cube_yp, goal_x, goal_y):
    eef = np.array([x, y])
    cube = np.array([cube_x, cube_y])
    cubep = np.array([cube_xp, cube_yp])
    goal = np.array([goal_x, goal_y])

    d_change = np.linalg.norm(cube - goal) - np.linalg.norm(cubep - goal)
    d_cube = euclidean(eef, cubep)
    d_goal = euclidean(cubep, goal)
    reward_change = 1000.0 * d_change # Too noisy!?
    reward_goal = 1000.0 * np.exp(-500 * d_goal)
    reward_cube = 100.0 * np.exp(-200 * d_cube)
    return reward_change + reward_goal + reward_cube


class Client():

    def __init__(self):
        self.arm = Arm()
        self.session = requests.Session()
        self.max_axis_move = 0.012
        # state is rotated and translated so that cube is in origin and
        # goal is on negative x-axis
        # state is eef relative to cube (2) + distance to cube to goal (1)
        critic = Critic(3, 2) # not needed for inference
        self.nn = Actor(3, 2, critic)
        try:
            self.nn.load_params('./ddpg/ddpg-actor-params.txt')
        except:
            logger.warning('Could not load parameters for network')
        sleep(2.0)

    def update_weights(self):
        logger.debug('Fetching new parameters')
        try:
            params = json.loads(self.session.get('http://beorn:5000/get_params').text)
        except requests.exceptions.ConnectionError:
            logger.error('Server not online? Could not update weights.')
            return
        self.nn.q.set_weights([np.array(param) for param in params])

    def start(self, demo=False):
        if demo:
            self.do_one_trial(noise_factor=0.1, max_movements=4096, goal=(0.0, 0.21))
        else:
            for i in range(16):
                if True:
                    #self.do_one_trial(noise_factor=0.5 + 0.5 * np.random.rand())
                    self.do_one_trial(noise_factor=0.1, max_movements=128)
                else:
                    trial = self.do_one_trial(noise_factor=0.0)
                    if trial is None:
                        continue
                    try:
                        headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
                        r = self.session.put('http://beorn:5000/put_test_trial', data=json.dumps(trial), headers=headers)
                        logger.debug('Sent test trial {}'.format(r))
                    except requests.exceptions.ConnectionError:
                        logger.error('Server not online? Could not send test trial.')

    def random_start_pose(self, cube_x, cube_y):
        sleep(1.0)
        theta = np.random.rand() * 2 * np.pi
        while True:
            x, y, z = (
                random_in_range(-0.10, 0.10),
                random_in_range(0.17, 0.25),
                0.03
            )
            if np.linalg.norm([x - cube_x, y - cube_y]) > 0.04:
                break
        self.arm.move_to(x, y, z, velocity=0.5)
        sleep(2.0)

    def next_move(self, eef_x, eef_y, cube_x, cube_y, goal_x, goal_y, state_vector, noise_factor=1.0):
        # new controls plus noise
        u = self.nn.u.predict(state_vector.reshape(1, 3))[0, :]
        u_noisy = (1 - noise_factor) * u + noise_factor * 2 * (np.random.rand(2) - 0.5)

        eef = np.array([eef_x, eef_y])
        cube = np.array([cube_x, cube_y])
        goal = np.array([goal_x, goal_y])

        d = cube - goal
        d_norm_before = np.linalg.norm(d)
        if d_norm_before > 0:
            alpha = np.arctan2(d[1], d[0])
            rot = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
            rel = np.dot(rot, u_noisy)
        else:
            g2e = eef - goal
            alpha = np.arctan2(g2e[1], g2e[0])
            rot = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
            rel = np.dot(rot, u)

        dx, dy = self.max_axis_move * rel
        return dx, dy

    def replace_cube(self, x, y):
        x_arm, y_arm, _ = self.arm.get_position()
        sleep(1.0)
        self.arm.move_to(x_arm, y_arm, 0.08, velocity=0.5) # raise arm first
        (x_now, y_now), _ = cube_pose_retry()
        sleep(1.0)
        while True:
            self.arm.move_to(x_now, y_now, 0.08)           # move to above cube
            sleep(1.0)
            (x_now, y_now), _ = cube_pose_retry()
            self.arm.move_to(x_now, y_now, 0.032)          # lower onto cube
            self.arm.set_pump(1)
            sleep(1.0)
            self.arm.move_to(x, y, 0.08, velocity=0.5)     # lift
            sleep(2.0)
            if not lw.cube_visible:
                break
        self.arm.move_to(x, y, 0.03, velocity=0.5)         # replace
        sleep(3.0)
        self.arm.set_pump(0)
        sleep(1.0)
        self.arm.move_to(x, y, 0.08, velocity=0.5)         # raise arm above cube
        sleep(1.0)

    def do_one_trial(self, noise_factor=1.0, max_movements=32, goal=None):
        logger.debug('Setting random start and goal poses')
        cube_start_x, cube_start_y, goal_x, goal_y = random_cube_start_goal()
        if goal:
            goal_x, goal_y = goal
        self.replace_cube(cube_start_x, cube_start_y)
        self.random_start_pose(cube_start_x, cube_start_y)

        experience = []
        #self.update_weights()
        logger.info('New goal at x: {}, y: {}'.format(goal_x, goal_y))
        for i in range(max_movements):
            x, y, _ = self.arm.get_position()
            (cube_x, cube_y), scans = cube_pose_retry()
            state = create_state_vector(x, y, cube_x, cube_y, goal_x, goal_y)
            dx, dy = self.next_move(x, y, cube_x, cube_y, goal_x, goal_y, state, noise_factor=noise_factor)
            dx_fixed, dy_fixed = remove_command_threshold(dx, dy, self.max_axis_move)
            logger.debug('Sending command: {:.3f}, {:.3f}. Corrected to: {:.3f} {:.3f}'.format(dx, dy, dx_fixed, dy_fixed))
            if self.arm._arm.is_connected():
                self.arm.move_to(x + dx_fixed, y + dy_fixed, 0.03)
            else:
                self.arm.disconnect()
                self.arm.stop()
                exit(-1)
            sleep(0.4)
            xp, yp, _ = self.arm.get_position()
            (cube_xp, cube_yp), scans_p = cube_pose_retry()
            error_euclid = euclidean([xp - x, yp - y], [dx, dy])
            if error_euclid > 0.01:
                logger.info('Ignoring transition, large command/measure error')
                continue
            state_prime = create_state_vector(xp, yp, cube_xp, cube_yp, goal_x, goal_y)
            r = reward(xp, yp, cube_x, cube_y, cube_xp, cube_yp, goal_x, goal_y)
            transition = {
                'x': list(state),
                'xp': list(state_prime),
                'real_x': [x, y, cube_x, cube_y, goal_x, goal_y],
                'real_xp': [xp, yp, cube_xp, cube_yp, goal_x, goal_y],
                'scans': list(scans),
                'scans_p': list(scans_p),
                'robot_id': os.environ['USER'],
                'u': [float(dx), float(dy)], # json complained
                'r': r,
            }
            if suspicious_transition(transition):
                logger.info('Ignoring transition, large deviation in cube pose')
                continue
            experience.append(transition)
            logger.info('Observed reward: {}'.format(r))
            if is_lose_pose(xp, yp, cube_xp, cube_yp):
                logger.info('Reached outside workspace')
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
lw = LidarWrapper()
sleep(2.0)
try:
    c.start(demo=('--demo' in sys.argv))
except KeyboardInterrupt:
    c.stop()
    exit(0)

c.stop()
lw.stop()

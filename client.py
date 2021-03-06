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
from pose_estimator import cube_pose, lw
from pyuarm.protocol import SERVO_BOTTOM, SERVO_LEFT, SERVO_RIGHT


logcolor.basic_config(level=logging.ERROR)
logger = logging.getLogger('client')
logger.setLevel(logging.INFO)


def suspicious_transition(X):
    ex1, ey1, cx1, cy1, gx1, gy1 = X['x']
    ex2, ey2, cx2, cy2, gx2, gy2 = X['xp']
    d = np.linalg.norm([cx1 - cx2, cy1 - cy2])
    return d > 0.03


def remove_command_threshold(dx, dy, max_axis_move):
    #dx += 0.0058 * np.sign(dx)
    #dy += 0.0058 * np.sign(dy)
    theta = np.arctan2(dy, dx)
    k = np.linalg.norm([dx, dy]) / max_axis_move
    dx += 0.0058 * np.cos(theta)
    dy += 0.0058 * np.sin(theta)
    return dx, dy


def create_state_vector(eef_x, eef_y, cube_x, cube_y, goal_x, goal_y):
    return np.array([[eef_x, eef_y, cube_x, cube_y, goal_x, goal_y]])


def random_in_range(a, b):
    return a + (b - a) * np.random.rand()


def is_lose_pose(x, y, cube_x, cube_y):
    if not (-0.12 <= x <= 0.12):
        return True
    if not (0.13 <= y <= 0.29):
        return True
    if np.linalg.norm([cube_x, cube_y - 0.21]) >= 0.08:
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
    for i in range(10):
        res = cube_pose()
        if res is None:
            sleep(1.0)
        else:
            return res


def random_cube_start_goal():
    theta = np.random.rand() * 2 * np.pi
    target_rotation = np.random.rand() * 2 * np.pi
    return (
        0.00 + 0.04 * np.cos(theta),
        0.21 + 0.025 * np.sin(theta),
        0.00 + 0.04 * np.cos(theta + np.pi),
        0.21 + 0.025 * np.sin(theta + np.pi)
    )


def reward(x, y, cube_x, cube_y, goal_x, goal_y):
    eef = np.array([x, y])
    cube = np.array([cube_x, cube_y])
    goal = np.array([goal_x, goal_y])

    d_cube = euclidean(eef, cube)
    d_goal = euclidean(cube, goal)
    reward_cube = np.exp(-1000 * d_cube ** 2) - 1
    reward_goal = 2 * (np.exp(-1000 * d_goal ** 2) - 1)
    return reward_cube + reward_goal


class Client():

    def __init__(self):
        self.arm = Arm()
        self.session = requests.Session()
        self.max_axis_move = 0.012
        # state is robot pose (2), cube relative pose (2), cube target pose (2)
        self.nn = NNet(x_size=2 + 2 + 2, u_size=2, mu_scaling=self.max_axis_move)
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
        for i in range(32):
            if i % 4:
                self.do_one_trial(noise_factor=1.0)
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

    def random_start_pose(self):
        sleep(1.0)
        theta = np.random.rand() * 2 * np.pi
        x, y, z = (
            0.00 + 0.10 * np.cos(theta),
            0.21 + 0.07 * np.sin(theta),
            0.03
        )
        self.arm.move_to(x, y, z, velocity=0.5)
        sleep(2.0)

    def next_move(self, state_vector, noise_factor=1.0):
        # new controls plus noise
        u_dx, u_dy = self.nn.mu.predict(state_vector)[0, :] + noise_factor * 0.01 * np.random.randn(2)
        if np.random.rand() < 0.3 * noise_factor:
            u_dx, u_dy = noise_factor * 0.01 * np.random.randn(2)

        euclid = np.sqrt(u_dx ** 2 + u_dy ** 2)
        if abs(u_dx) > self.max_axis_move:
            u_dx = self.max_axis_move * np.sign(u_dx)
        if abs(u_dy) > self.max_axis_move:
            u_dy = self.max_axis_move * np.sign(u_dy)

        return u_dx, u_dy

    def replace_cube(self, x, y):
        x_arm, y_arm, _ = self.arm.get_position()
        sleep(1.0)
        self.arm.move_to(x_arm, y_arm, 0.08) # raise arm first
        x_now, y_now, _ = cube_pose_retry()
        sleep(1.0)
        self.arm.move_to(x_now, y_now, 0.08)               # move to above cube
        sleep(1.0)
        x_now, y_now, _ = cube_pose_retry()
        self.arm.move_to(x_now, y_now, 0.032)              # lower onto cube
        self.arm.set_pump(1)
        sleep(1.0)
        self.arm.move_to(x, y, 0.08)                       # lift
        sleep(1.0)
        self.arm.move_to(x, y, 0.03, velocity=0.5)         # replace
        sleep(3.0)
        self.arm.set_pump(0)
        sleep(1.0)
        self.arm.move_to(x, y, 0.08)         # raise arm above cube
        sleep(1.0)

    def do_one_trial(self, noise_factor=1.0, max_movements=32):
        logger.debug('Setting random start and goal poses')
        cube_start_x, cube_start_y, goal_x, goal_y = random_cube_start_goal()
        self.replace_cube(cube_start_x, goal_y)
        self.random_start_pose()

        experience = []
        self.update_weights()
        logger.info('New goal at x: {}, y: {}'.format(goal_x, goal_y))
        for i in range(max_movements):
            x, y, _ = self.arm.get_position()
            cube_x, cube_y, _ = cube_pose_retry()
            state = create_state_vector(x, y, cube_x, cube_y, goal_x, goal_y)
            dx, dy = self.next_move(state, noise_factor=noise_factor)
            dx_fixed, dy_fixed = remove_command_threshold(dx, dy, self.max_axis_move)
            logger.debug('Sending command: {:.3f}, {:.3f}. Corrected to: {:.3f} {:.3f}'.format(dx, dy, dx_fixed, dy_fixed))
            if self.arm._arm.is_connected():
                self.arm.move_to(x + dx_fixed, y + dy_fixed, 0.03)
            else:
                self.arm.disconnect()
                self.arm.stop()
                exit(-1)
            sleep(0.2)
            xp, yp, _ = self.arm.get_position()
            cube_xp, cube_yp, _ = cube_pose_retry()
            error_euclid = euclidean([xp - x, yp - y], [dx, dy])
            if error_euclid > 0.01:
                logger.info('Ignoring transition, large command/measure error')
                continue
            state_prime = create_state_vector(xp, yp, cube_xp, cube_yp, goal_x, goal_y)
            r = reward(xp, yp, cube_xp, cube_yp, goal_x, goal_y)
            if is_lose_pose(xp, yp, cube_xp, cube_yp):
                r = -4
            transition = {
                'x': list(state[0, :]),
                'xp': list(state_prime[0, :]),
                'u': [dx, dy],
                'r': r,
            }
            if suspicious_transition(transition):
                logger.info('Ignoring transition, large deviation in cube pose')
                continue
            experience.append(transition)
            logger.info('Observed reward: {}'.format(r))
            logger.debug('Cube at: {}'.format(state_prime[0, 2:4]))
            if is_lose_pose(xp, yp, cube_xp, cube_yp):
                logger.info('Reached outside workspace')
                break
            if is_win_pose(xp, yp, cube_xp, cube_yp, goal_x, goal_y):
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
    exit(0)

c.stop()
lw.stop()

import os
import pickle
import threading
from time import sleep
from copy import deepcopy
from queue import Queue, Empty

import pyuarm
import numpy as np
from scipy.spatial.distance import euclidean
from pyuarm.protocol import SERVO_BOTTOM, SERVO_LEFT, SERVO_RIGHT, SERVO_HAND

SERVOS = [SERVO_BOTTOM, SERVO_LEFT, SERVO_RIGHT, SERVO_HAND]

def to_deg(rad):
    return 180 * rad / np.pi

def to_rad(deg):
    return np.pi * deg / 180.0

k1 = OF = 0.107
k2 = FA = 0.02
k3 = AB = 0.148
k4 = BC = 0.16
k5 = CE = 0.035
k6 = DE = 0.06

def forward(alpha, beta, gamma):
    r = k2 + k3 * np.cos(alpha) + k4 * np.cos(beta) + k5
    z = k1 + k3 * np.sin(alpha) + k4 * np.sin(beta) - k6
    return r * np.cos(gamma), r * np.sin(gamma), z

def inverse(x, y, z):
    # Polar coordinates, solve in vertical 2D plane
    gamma = np.arctan2(y, x)
    r = np.sqrt(x ** 2 + y ** 2) - k2 - k5
    # Calculate from first joint to end of forearm
    z_p = z - k1 + k6
    alpha_part = np.arctan2(z_p, r)
    # Rotate so triangle base joint, next joint, end of forearm "lies down"
    r_pp = r * np.cos(-alpha_part) - z_p * np.sin(-alpha_part)
    z_pp = r * np.sin(-alpha_part) + z_p * np.cos(-alpha_part)
    alpha_rest = np.arccos((k3 ** 2 + r_pp ** 2 - k4 ** 2) / (2 * k3 * r_pp))
    alpha = alpha_part + alpha_rest
    top_angle = np.arccos((k3 ** 2 - r_pp ** 2 + k4 ** 2) / (2 * k3 * k4))
    beta = np.arctan2(z_p - k3 * np.sin(alpha), r - k3 * np.cos(alpha))
    return alpha, beta, gamma

UPDATE_FREQ = 128 # Hz

class Arm():
    
    def __init__(self, arm=None):
        self._target = None
        self._cancel_worker = False
        if arm is None:
            self._arm = pyuarm.get_uarm()
        else:
            self._arm = arm
        home = os.environ.get('HOME')
        with open(os.path.join(home, 'uarm_params.pkl'), 'rb') as f:
            self._uarm_params = pickle.load(f)
        self._queue = []
        self._worker_thread = threading.Thread(target=self._worker)
        self._worker_thread.start()
        
    def disconnect(self):
        self._arm.disconnect()
        
    def stop(self):
        self._cancel_worker = True
        
    def cartesian_to_servo_angles(self, x, y, z):
        alpha_ideal, beta_ideal, gamma_ideal = map(to_deg, inverse(x, y, z))
        ideal_real = self._uarm_params['ideal_real']
        return (
            ideal_real[SERVO_LEFT]['slope'] * alpha_ideal + ideal_real[SERVO_LEFT]['intercept'],
            ideal_real[SERVO_RIGHT]['slope'] * beta_ideal + ideal_real[SERVO_RIGHT]['intercept'],
            ideal_real[SERVO_BOTTOM]['slope'] * gamma_ideal + ideal_real[SERVO_BOTTOM]['intercept'],
        )
    
    def servo_angles_to_cartesian(self, alpha, beta, gamma):
        ideal_real = self._uarm_params['ideal_real']
        alpha_p = to_rad((alpha - ideal_real[SERVO_LEFT]['intercept']) / ideal_real[SERVO_LEFT]['slope'])
        beta_p = to_rad((beta - ideal_real[SERVO_RIGHT]['intercept']) / ideal_real[SERVO_RIGHT]['slope'])
        gamma_p = to_rad((gamma - ideal_real[SERVO_BOTTOM]['intercept']) / ideal_real[SERVO_BOTTOM]['slope'])
        return forward(alpha_p, beta_p, gamma_p)
    
    def servo_angles_to_command(self, alpha, beta, gamma):
        km = self._uarm_params['commanded_measured']
        return (
            (alpha - km[SERVO_LEFT]['intercept']) / km[SERVO_LEFT]['slope'],
            (beta - km[SERVO_RIGHT]['intercept']) / km[SERVO_RIGHT]['slope'],
            (gamma - km[SERVO_BOTTOM]['intercept']) / km[SERVO_BOTTOM]['slope'],
        )
        
    def get_position(self):
        return self.servo_angles_to_cartesian(
            self._arm.get_servo_angle(SERVO_LEFT),
            self._arm.get_servo_angle(SERVO_RIGHT),
            self._arm.get_servo_angle(SERVO_BOTTOM),
        )
    
    def _set_angles(self, alpha, beta, gamma):
        self._arm.set_servo_angle(SERVO_LEFT, alpha)
        self._arm.set_servo_angle(SERVO_RIGHT, beta)
        self._arm.set_servo_angle(SERVO_BOTTOM, gamma)
        
    def _set_cartesian(self, x, y, z):
        a = self.cartesian_to_servo_angles(x, y, z)
        b = self.servo_angles_to_command(*a)
        self._set_angles(*b)
        
    def set_pump(on):
        self._arm.set_pump(on)
        
    def move_to(self, x, y, z, duration=None, velocity=None):
        """
        x, y, z : float
            meter
        duration : float
            seconds
        velocity : float
            m / s, ignored if duration is specified
        """
        # add extreme safety bounds here!
        
        if duration is None and velocity is None:
            self._set_cartesian(x, y, z)
            return
        
        if self._queue:
            self._queue = []
            sleep(0.5)
        
        a = np.array(self.get_position())
        b = np.array([x, y, z])
        distance = euclidean(a, b)
        if velocity is not None:
            duration = distance / velocity
            
        # invariant: we have a, b, and duration
        n_pieces = UPDATE_FREQ * duration
        for i in range(1, int(n_pieces + 1)):
            partial_dist = i * distance / n_pieces
            x = a + (b - a) * i / n_pieces
            self._queue.append((x[0], x[1], x[2]))
        return
    
    def _worker(self):
        while not self._cancel_worker:
            try:
                x, y, z = self._queue[0]
                self._queue = self._queue[1:]
                self._set_cartesian(x, y, z)
            except IndexError:
                pass
            sleep(1.0 / UPDATE_FREQ)

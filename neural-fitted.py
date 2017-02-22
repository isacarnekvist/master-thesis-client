from __future__ import print_function

import sys
import json
import pickle
import threading
from operator import mul
from copy import deepcopy
from functools import reduce
from datetime import datetime, timedelta

import keras
import theano
import theano.tensor as T
from keras import backend as K
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dense, Merge, Input, Lambda, merge, Layer, BatchNormalization

import numpy as np

from naf.priority_buffer import PriorityBuffer
from environment import Environment, WIN, LOSE, NEUTRAL

theano.sandbox.cuda.use('gpu0')

MAX_DIST = 0.01

print('Creating replay buffer')
environment = Environment(MAX_DIST)
replay_buffer = PriorityBuffer(2 ** 10)

gx, gy = environment.goal_x, environment.goal_y

print('Filling up replay buffer')
n_trials = 4094 * 4
n_trials = 4
for trial in range(n_trials):
    print('\r' * 12, end='')
    print('{}/{}'.format(trial + 1, n_trials), end='')
    environment.reset()
    if np.random.rand() < 0.8:
        tmp_gx = gx
        tmp_gy = gy
    else:
        tmp_gx = 0.30 * np.random.rand() - 0.15
        tmp_gy = 0.20 * np.random.rand() - 0.12
    for i in range(32):
        environment.goal_x = tmp_gx
        environment.goal_y = tmp_gy
        if np.random.rand() < 0.8:
            mu = environment.heuristic_move()
        else:
            mu = np.random.randn(2) * 0.005
        environment.goal_x = gx
        environment.goal_y = gy
        x1 = environment.get_state()
        s, r, x2 = environment.interact(*mu)
        replay_buffer.add({
            'x1': x1,
            'x2': x2,
            'u': mu,
            'r': r
        }).set_value(10.0)
        if s in [WIN, LOSE]:
            break
print()
print(replay_buffer)


import keras
import theano
import numpy as np
import theano.tensor as T
from keras import backend as K
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.models import Sequential, Model
from keras.layers import Dense, Merge, Input, Lambda, BatchNormalization


class DDPG:

    def predict(self, *x):
        return self.nn.predict(list(x))
    
    @property
    def trainable_weights(self):
        return [w for w in self.nn.trainable_weights if not w.name.startswith('bn')]
    
    def soft_update(self, weights, lr=0.001):
        """
        Accepts theano tensors as inputs
        """
        for w_old, w_new in zip(self.nn.weights, weights):
            w_old.set_value(
                lr * w_new.get_value() + (1 - lr) * w_old.get_value()
            )
            

class Critic(DDPG):
    
    def __init__(self, x_size, u_size, hidden_size=100):
    
        x = Input(shape=(x_size, ), name='x')
        u = Input(shape=(u_size, ), name='u')
        x_model = Model(input=x, output=x)
        u_model = Model(input=u, output=u)
        
        first_part = Sequential([
            BatchNormalization(input_shape=(x_size,), name='bn1'),
            Dense(output_dim=hidden_size, activation='relu', name='fc1', W_regularizer=l2(0.01)),
        ])
        
        self.nn = Sequential([
            Merge([first_part, u_model], mode='concat'),
            BatchNormalization(name='bn2'),
            Dense(output_dim=hidden_size - 100, activation='relu', name='fc2', W_regularizer=l2(0.01)),
            BatchNormalization(name='bn3'),
            Dense(output_dim=(1), name='Q', W_regularizer=l2(0.01)),
        ])

        adam = Adam(lr=0.0001)
        self.nn.compile(loss='mse', optimizer=adam)
        self._gradients = theano.function(
            self.nn.inputs + [K.learning_phase()],
            T.grad(self.nn.output[0, 0], u_model.output),
            allow_input_downcast=True
        )

    def gradients(self, x, u):
        assert x.shape[0] == 1
        return self._gradients(x, u, False)
        
    
class Actor(DDPG):
    
    def __init__(self, u_size, max_dist, q):
        self.u_size = 2
        self.max_dist = max_dist
        self.q = q
        
    def predict(self, x, n_samples=64):
        np.random.seed(1)
        n_samples = n_samples
        batch_size = x.shape[0]
        state_size = x.shape[1]
        mu = np.zeros((batch_size, self.u_size))
        for i in range(batch_size):
            x_repeat = np.tile(x[i:i+1, :], n_samples).reshape((n_samples, state_size))
            q_max = -np.inf
            u = MAX_DIST * (2 * np.random.rand(n_samples, 2) - 1)
            q_vals = self.q.predict(x_repeat, u)
            mu[i, :] = u[np.argmax(q_vals)]
        return mu


print('Compiling networks')
hidden_size = 400
critic = Critic((2 + 2 + 2), 2, hidden_size=hidden_size)
critic_target = Critic((2 + 2 + 2), 2, hidden_size=hidden_size)
critic.nn.set_weights(critic_target.nn.get_weights())
actor = Actor(2, MAX_DIST, critic)
actor_target = Actor(2, MAX_DIST, critic_target)
try:
    with open('critic_params.json', 'w') as f:
        p = map(np.array, json.loads(f.read()))
        critic.nn.set_weights(p)
    with open('critic_target_params.json', 'w') as f:
        p = map(np.array, json.loads(f.read()))
        critic_target.nn.set_weights(p)
except Exception as e:
    print('Error loading params:', e)


def return_average(actor, gamma=0.98):
    rewards = []
    for trial in range(8):
        np.random.seed(trial)
        environment.reset()
        n_steps = 256
        return_ = 0.0
        for i in range(n_steps):
            mu = actor.predict(environment.get_state())
            _, r, _ = environment.interact(*mu.flatten())
            return_ += gamma ** i * r
        rewards.append(return_)
    return np.mean(rewards), np.std(rewards)


def fit():
    epoch_size = 4096
    batch_size = 64
    gamma = 0.98
    epsilon = 0.1

    reward_averages = []
    reward_averages_target = []

    X = np.zeros((epoch_size, 6))
    Xp = np.zeros((epoch_size, 6))
    U = np.zeros((epoch_size, 2))
    R = np.zeros((epoch_size, 1))

    n_iterations = 2048.0
    latest_info = datetime.now() - timedelta(seconds=30)
    latest_trial_plot = datetime.now() - timedelta(seconds=60)
    a = 0
    for a in range(a, int(n_iterations)):
        print('iteration {} / {}'.format(a + 1, n_iterations))
        environment.reset()
        latest_trial = []
        latest_rewards = []
        for b in range(epoch_size):
            x1 = environment.get_state()
            mu = actor.predict(x1)
            noise = np.random.randn(1, 2) * 0.05 * (4 * n_iterations - 3 * a) / (4 * n_iterations)

            state, reward, x2 = environment.interact(*(mu)[0, :])
            latest_trial.append(x2[0, :])
            latest_rewards.append(reward)
            replay_buffer.add({
                'x1': x1,
                'x2': x2,
                'u': mu,
                'r': reward
            }).set_value(10.0)
            if state in [LOSE, WIN] or b == epoch_size - 1 or len(latest_trial) % 32 == 0:
                latest_trial = []
                latest_rewards = []
                environment.reset()
        
        n_inner = 32
        for i in range(n_inner):
            exp_nodes = []
            for b in range(epoch_size):
                sample = replay_buffer.sample()
                exp_nodes.append(sample)
                X[b, :] = sample.data['x1']
                Xp[b, :] = sample.data['x2']
                R[b, :] = sample.data['r']
                U[b, :] = sample.data['u']
                
            Q = critic.predict(X, U)
            Y = R + gamma * critic_target.predict(Xp, actor_target.predict(Xp))
            [node.set_value(abs(delta) + epsilon) for node, delta in zip(exp_nodes, (Q - Y)[:, 0])]
            beta = 1 / (1 + np.exp(- 10 * a / n_iterations + 6))
            sample_weight = np.array([1.0 / node.value for node in exp_nodes]) ** beta

            timestamp = datetime.now()
            critic.nn.fit([X, U], Y, verbose=0, sample_weight=sample_weight, batch_size=batch_size, nb_epoch=16)
            print('fit() took {}'.format(datetime.now() - timestamp))
            
            critic_target.soft_update(critic.nn.weights, lr=0.001)

            if datetime.now() > latest_info + timedelta(seconds=30):
                r_avg, r_std = return_average(actor)
                print('r_avg: {:.4f} r_std: {:.4f} beta: {:.4f} {}'.format(r_avg, r_std, beta, replay_buffer))
                latest_info = datetime.now()
                with open('critic_params.json', 'w') as f:
                    params = list(map(lambda x: x.tolist(), critic.nn.get_weights()))
                    f.write(json.dumps(params))
                with open('critic_target_params.json', 'w') as f:
                    params = list(map(lambda x: x.tolist(), critic_target.nn.get_weights()))
                    f.write(json.dumps(params))
                


if __name__ == '__main__':
    fit()

from __future__ import print_function

import sys
import pickle
import threading
from operator import mul
from copy import deepcopy
from functools import reduce
from datetime import datetime, timedelta

if sys.version_info.major == 2:
    from Queue import Queue
else:
    from queue import Queue

import keras
import theano
import numpy as np
import theano.tensor as T
from keras import backend as K
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dense, Merge, Input, Lambda, merge, Layer, BatchNormalization


from naf.priority_buffer import PriorityBuffer


theano.sandbox.cuda.use('gpu0')


WIN = 0
LOSE = 1
NEUTRAL = 2
MAX_DIST = 0.01


def create_state_vector(eef_x, eef_y, circle_x, circle_y, goal_x, goal_y):
    return np.array([
        [eef_x, eef_y, circle_x, circle_y, goal_x, goal_y]
    ], dtype=np.float32)


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
        
        self.nn = Sequential([
            Merge([x_model, u_model], mode='concat'),
            BatchNormalization(input_shape=(x_size,), name='bn1'),
            Dense(output_dim=hidden_size, activation='relu', name='fc1'),
            BatchNormalization(name='bn2'),
            Dense(output_dim=hidden_size, activation='relu', name='fc2'),
            BatchNormalization(name='bn3'),
            Dense(output_dim=(1), name='Q'),
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
    
    def __init__(self, x_size, u_size, mu_scaling, hidden_size=100):
        x = Input(shape=(x_size, ), name='state')
        self.nn = Sequential([
            BatchNormalization(input_shape=(x_size,), name='bn1'),
            Dense(input_shape=(x_size,), output_dim=hidden_size, activation='relu', name='fc1'),
            BatchNormalization(name='bn2'),
            Dense(output_dim=hidden_size, activation='relu', name='fc2'),
            BatchNormalization(name='bn3'),
            Dense(output_dim=u_size, name='mu_unscaled', activation='tanh'),
            Lambda(lambda x: mu_scaling * x, output_shape=(u_size, ), name='mu')
        ])
        
        # This optimizer won't be needed, learning from policy gradient
        self.nn.compile(loss='mse', optimizer='sgd')
        
        # gradients
        params = self.trainable_weights
        gradients = [T.grad(self.nn.output[0, i], params) for i in range(u_size)]
        gradients_list = []
        for g in gradients:
            gradients_list.extend(g)
        self._gradients = theano.function(
            self.nn.inputs + [K.learning_phase()],
            gradients_list,
            allow_input_downcast=True
        )
    
    def gradients(self, x):
        assert x.shape[0] == 1
        res = []
        for g in self._gradients(x, False):
            param_len = reduce(mul, g.shape)
            res.extend(np.array(g.reshape(param_len, )))
        return np.array(res).reshape((2, int(len(res) / 2)))
    
    def update_with_policy_gradient(self, policy_gradient, lr=0.0001):
        """
        Update from separate actor and critic gradients, which
        multiply to make the policy gradient
        """
        i = 0
        policy_gradient = policy_gradient.astype(np.float32)
        for g in self.trainable_weights:
            v = g.get_value()
            param_len = reduce(mul, v.shape)
            g.set_value(v + lr * policy_gradient[0, i:i + param_len].reshape(v.shape))
            i += param_len


class Circle:
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = 0.02
        
    def interact(self, x, y):
        theta = np.arctan2(y - self.y, x - self.x)
        center_distance = np.linalg.norm([self.y - y, self.x - x])
        distance = self.radius - center_distance
        if center_distance > self.radius:
            return
        self.x -= distance * np.cos(theta)
        self.y -= distance * np.sin(theta)
        
class Environment:
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        # Random on inner and outer circle
        eef_theta = np.random.rand() * 2 * np.pi
        self.eef_x = 0.10 * np.cos(eef_theta)
        self.eef_y = 0.20 + 0.07 * np.sin(eef_theta)
        circle_theta = np.random.rand() * 2 * np.pi
        circle_x = 0.04 * np.cos(circle_theta)
        circle_y = 0.20 + 0.02 * np.sin(circle_theta)
        self.circle = Circle(circle_x, circle_y)
        while True:
            goal_theta = np.random.rand() * 2 * np.pi
            self.goal_x = 0.04 * np.cos(goal_theta)
            self.goal_y = 0.20 + 0.02 * np.sin(goal_theta)
            if np.linalg.norm([self.goal_x - circle_x, self.goal_y - circle_y]) > 0.04:
                break
        while True:
            self.eef_x  = -0.10 + np.random.rand() * 0.20
            self.eef_y  =  0.12 + np.random.rand() * 0.17
            if np.linalg.norm([self.eef_x - circle_x, self.eef_y - circle_y]) < 0.04:
                continue
            else:
                break

    def get_state(self):
        return create_state_vector(
            self.eef_x,
            self.eef_y,
            self.circle.x,
            self.circle.y,
            self.goal_x,
            self.goal_y,
        )

    def interact(self, dx, dy):
        dist = np.linalg.norm([dx, dy])
        if dist > MAX_DIST:
            dx = MAX_DIST * dx / dist
            dy = MAX_DIST * dy / dist
        self.eef_x += dx
        self.eef_y += dy
        self.circle.interact(self.eef_x, self.eef_y)
        state = NEUTRAL
        reward = -4
        if not -0.15 <= self.eef_x <= 0.15:
            state = LOSE
        elif not 0.10 <= self.eef_y <= 0.30:
            state = LOSE
        elif not -0.15 <= self.circle.x <= 0.15:
            state = LOSE
        elif not 0.10 <= self.circle.y <= 0.30:
            state = LOSE
        elif np.linalg.norm([self.goal_x - self.circle.x, self.goal_y - self.circle.y]) < 0.005:
            state = WIN
            
        if state != LOSE:
            eef2circle = np.linalg.norm([self.eef_x - self.circle.x, self.eef_y - self.circle.y])
            circle2goal = np.linalg.norm([self.goal_x - self.circle.x, self.goal_y - self.circle.y])
            reward = (
                np.exp(-200 * eef2circle ** 2) - 1 +
                2 * np.exp(-200 * circle2goal ** 2) - 1
            )
        
        return state, reward, self.get_state()


def last_reward_average(actor):
    rewards = []
    for trial in range(64):
        np.random.seed(trial)
        e.reset()
        n_steps = 128
        for i in range(n_steps):
            mu = actor.predict(e.get_state())
            _, r, _ = e.interact(*mu.flatten())
        rewards.append(r)
    return np.mean(rewards), np.std(rewards)


e = Environment()

print('Compiling networks')
hidden_size = 200
actor = Actor((2 + 2 + 2), 2, MAX_DIST, hidden_size=hidden_size)
actor_target = Actor((2 + 2 + 2), 2, MAX_DIST, hidden_size=hidden_size)
actor_target.nn.set_weights(actor.nn.get_weights())

critic = Critic((2 + 2 + 2), 2, hidden_size=hidden_size)
critic_target = Critic((2 + 2 + 2), 2, hidden_size=hidden_size)
critic_target.nn.set_weights(critic.nn.get_weights())

#print('Loading saved params')
#nets = [actor, actor_target, critic, critic_target]
#names = ['actor', 'actor_target', 'critic', 'critic_target']
#for net, name in zip(nets, names):
#    with open(name + '.pkl', 'rb') as f:
#        p = pickle.load(f)
#        net.nn.set_weights(p)
#
#print('Loading replay buffer')
#try:
#    with open('replay_buffer.pkl', 'rb') as f:
#        replay_buffer = pickle.load(f)
#except Exception as e:
#    print('Failed loading replay buffer, creating new.', e)
#    replay_buffer = PriorityBuffer(2 ** 21)
replay_buffer = PriorityBuffer(2 ** 20)

epoch_size = 1024
batch_size = 32
gamma = 0.98
epsilon = 0.1

X = np.zeros((epoch_size, 6))
Xp = np.zeros((epoch_size, 6))
U = np.zeros((epoch_size, 2))
R = np.zeros((epoch_size, 1))
gradient_len = actor.gradients(X[:1, :]).shape[1]
policy_gradient = np.zeros((1, gradient_len))

n_iterations = 2048.0
latest_info = datetime.now() - timedelta(seconds=30)
latest_param_save = datetime.now()
a = 0
for a in range(a, int(n_iterations)):
    print('Iteration {} / {}'.format(a + 1, int(n_iterations)))
    e.reset()
    latest_trial = []
    for b in range(epoch_size):
        x1 = e.get_state()
        mu = actor.predict(x1)
            
        noise = np.random.randn(1, 2) * MAX_DIST * (4.0 - 3.0 * a / n_iterations) / 4.0
        mu = mu + noise
        dist = np.linalg.norm(mu)
        if dist > MAX_DIST:
            mu = mu * MAX_DIST / dist
        state, reward, x2 = e.interact(*(mu)[0, :])
        latest_trial.append(x2[0, :])
        replay_buffer.add({
            'x1': x1,
            'x2': x2,
            'u': mu,
            'r': reward
        }).set_value(10.0)
        if state in [LOSE, WIN] or len(latest_trial) > 128:
            latest_trial = []
            e.reset()
    
    n_inner = 16
    for i in range(n_inner):
        exp_nodes = []
        gen_targets_start = datetime.now()
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
        beta = np.exp((a - n_iterations) / (0.1 * n_iterations))
        sample_weight = np.array([1.0 / node.value for node in exp_nodes]) ** beta
        print('Target generation {} took {}'.format(epoch_size, datetime.now() - gen_targets_start))

        train_start = datetime.now()
        critic.nn.fit([X, U], Y, verbose=0, sample_weight=sample_weight, batch_size=batch_size, nb_epoch=16)
        print('fit() needed {}'.format(datetime.now() - train_start))

        train_start = datetime.now()
        ls_grads = []
        rs_grads = []
        for b in range(epoch_size):
            ls_grads.append(critic.gradients(X[b:b + 1, :], U[b:b + 1, :]))
            rs_grads.append(actor.gradients(X[b:b + 1, :]))
        print('policy partial gradients needed {}'.format(datetime.now() - train_start))

        train_start = datetime.now()
        policy_gradient *= 0
        for b in range(epoch_size):
            policy_gradient += sample_weight[b] * np.dot(
                ls_grads[b],
                rs_grads[b]
            ) / epoch_size
        print('policy gradient average and dots needed {}'.format(datetime.now() - train_start))

        train_start = datetime.now()
        actor.update_with_policy_gradient(policy_gradient, lr=0.1)
        print('policy gradient update needed {}'.format(datetime.now() - train_start))

        train_start = datetime.now()
        actor_target.soft_update(actor.nn.weights, lr=0.001)
        critic_target.soft_update(critic.nn.weights, lr=0.001)
        print('soft updates needed {}'.format(datetime.now() - train_start))
        
        if datetime.now() > latest_info + timedelta(seconds=30):
            last_reward_avg, last_reward_std = last_reward_average(actor)
            print('last reward avg: {:.3f} std: {:.3f} beta: {:.3f} outer: {}/{} inner: {}/{} {}'.format(
                last_reward_avg, last_reward_std, beta, a, n_iterations, i, n_inner, replay_buffer
            ))
            latest_info = datetime.now()

        #if datetime.now() > latest_param_save + timedelta(seconds=5 * 60):
        #    print('Saving Parameters')
        #    nets = [actor, actor_target, critic, critic_target]
        #    names = ['actor', 'actor_target', 'critic', 'critic_target']
        #    for net, name in zip(nets, names):
        #        with open(name + '.pkl', 'wb') as f:
        #            pickle.dump(net.nn.get_weights(), f, protocol=2)
        #    print('Saving replay buffer')
        #    with open('replay_buffer.pkl', 'wb') as f:
        #        pickle.dump(replay_buffer, f, protocol=2)
        #    latest_param_save = datetime.now()

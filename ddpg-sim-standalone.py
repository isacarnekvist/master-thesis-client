from __future__ import print_function

import sys
import json
import pickle
import threading
from operator import mul
from copy import deepcopy
from functools import reduce
from datetime import datetime, timedelta
from multiprocessing import Process, Queue, Value, Pool
if sys.version_info.major == 2:
    from Queue import Empty
else:
    from queue import Empty

import keras
import theano
import theano.tensor as T
from keras import backend as K
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dense, Merge, Input, Lambda, merge, Layer, BatchNormalization

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#from ddpg import Actor, Critic
from naf.priority_buffer import PriorityBuffer
from environment import Environment, WIN, LOSE, NEUTRAL


e = Environment(0.01)
replay_buffer = PriorityBuffer(2 ** 20)

rewards = []
for trials in range(4096 * 2):
    e.reset()
    for i in range(32):
        mu = e.heuristic_move()
        x1 = e.get_state()
        s, r, x2 = e.interact(*mu)
        replay_buffer.add({
            'x1': x1,
            'x2': x2,
            'u': mu,
            'r': r
        }).set_value(10.0)
        if s == WIN:
            break
        
from operator import mul
from functools import reduce

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
    
    def __init__(self, x_size, u_size, mu_scaling, hidden_size=100):
        
        # for Adam
        self.t = 0
        self.alpha = 0.001
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8

        x = Input(shape=(x_size, ), name='state')
        self.nn = Sequential([
            BatchNormalization(input_shape=(x_size,), name='bn1'),
            Dense(input_shape=(x_size,), output_dim=hidden_size, activation='relu', name='fc1'),
            BatchNormalization(name='bn2'),
            Dense(output_dim=hidden_size - 100, activation='relu', name='fc2'),
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
            res.extend(g.flatten())
        return np.array(res).reshape((2, int(len(res) / 2)))
    
    def update_with_policy_gradient(self, policy_gradient):
        """
        Update from separate actor and critic gradients, which
        multiply to make the policy gradient
        """
        i = 0
        if self.t == 0:
            self.m = np.zeros(policy_gradient.shape)
            self.v = np.zeros(policy_gradient.shape)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * policy_gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * policy_gradient ** 2
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        policy_gradient = policy_gradient.astype(np.float32)
        for g in self.trainable_weights:
            prev = g.get_value()
            param_len = reduce(mul, prev.shape)
            mh = m_hat[0, i:i + param_len].reshape(prev.shape).astype(np.float32)
            vh = v_hat[0, i:i + param_len].reshape(prev.shape).astype(np.float32)
            g.set_value(prev + self.alpha * mh / (np.sqrt(vh) + self.epsilon))
            i += param_len


MAX_DIST = 0.01

def return_average(actor, gamma=0.98):
    rewards = []
    for trial in range(8):
        np.random.seed(trial)
        e.reset()
        n_steps = 256
        return_ = 0.0
        for i in range(n_steps):
            mu = actor.predict(e.get_state())
            _, r, _ = e.interact(*mu.flatten())
            return_ += gamma ** i * r
        rewards.append(return_)
    return np.mean(rewards), np.std(rewards)


def gradient_calculator(stop_flag, actor, critic, params_q, shared_state_action_q, shared_results_q):
    while not stop.value:
        try:
            if not params_q.empty():
                actor_params, critic_params = params_q.get()
                actor.nn.set_weights(actor_params)
                critic.nn.set_weights(critic_params)
            try:
                sample_weight, x, u = shared_state_action_q.get(timeout=1.0)
            except Empty:
                continue
            a_grad = actor.gradients(x)
            c_grad = critic.gradients(x, u)
            shared_results_q.put(sample_weight * np.dot(c_grad, a_grad))
        except KeyboardInterrupt:
            return


print('Compiling networks')
hidden_size = 200
actor = Actor((2 + 2 + 2), 2, MAX_DIST, hidden_size=hidden_size)
actor_target = Actor((2 + 2 + 2), 2, MAX_DIST, hidden_size=hidden_size)
actor_target.nn.set_weights(actor.nn.get_weights())

critic = Critic((2 + 2 + 2), 2, hidden_size=hidden_size)
critic_target = Critic((2 + 2 + 2), 2, hidden_size=hidden_size)
critic_target.nn.set_weights(critic.nn.get_weights())


print('Starting gradient workers')
n_gradient_workers = 4
stop = Value('b', False)

processes = []
params_qs = []
shared_results_q = Queue(1024)
shared_state_action_q = Queue(1024)

for n in range(n_gradient_workers):
    param_q = Queue(4)
    p = Process(
        target=gradient_calculator,
        args=(stop, actor, critic, param_q, shared_state_action_q, shared_results_q)
    )
    p.start()
    processes.append(p)
    params_qs.append(param_q)

epoch_size = 1024
batch_size = 32
gamma = 0.98
epsilon = 0.1

reward_averages = []
reward_averages_target = []

X = np.zeros((epoch_size, 6))
Xp = np.zeros((epoch_size, 6))
U = np.zeros((epoch_size, 2))
R = np.zeros((epoch_size, 1))
gradient_len = actor.gradients(X[:1, :]).shape[1]
policy_gradient = np.zeros((1, gradient_len))

def fit():
    n_iterations = 2048.0
    latest_plot = datetime.now() - timedelta(seconds=30)
    latest_trial_plot = datetime.now() - timedelta(seconds=60)
    a = 0
    for a in range(a, int(n_iterations)):
        print('iteration {} / {}'.format(a + 1, n_iterations))
        #e.reset()
        #latest_trial = []
        #latest_rewards = []
        #for b in range(epoch_size):
        #    x1 = e.get_state()
        #    mu = actor.predict(x1, epsilon=1.0 * (n_iterations - a) / n_iterations + 0.1 * a / n_iterations)
        #        
        #    state, reward, x2 = e.interact(*(mu)[0, :])
        #    latest_trial.append(x2[0, :])
        #    latest_rewards.append(reward)
        #    replay_buffer.add({
        #        'x1': x1,
        #        'x2': x2,
        #        'u': mu,
        #        'r': reward
        #    }).set_value(10.0)
        #    if state in [LOSE, WIN] or b == epoch_size - 1 or len(latest_trial) % 32 == 0:
        #        latest_trial = []
        #        latest_rewards = []
        #        e.reset()
        
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
                
            timestamp = datetime.now()
            Q = critic.predict(X, U)
            Y = R + gamma * critic_target.predict(Xp, actor_target.predict(Xp))
            [node.set_value(abs(delta) + epsilon) for node, delta in zip(exp_nodes, (Q - Y)[:, 0])]
            beta = np.exp((a - n_iterations) / (0.1 * n_iterations))
            sample_weight = np.array([1.0 / node.value for node in exp_nodes]) ** beta
            print('batch prepared, took {}'.format(datetime.now() - timestamp))

            timestamp = datetime.now()
            critic.nn.fit([X, U], Y, verbose=0, sample_weight=sample_weight, batch_size=batch_size, nb_epoch=4)
            print('fit() took {}'.format(datetime.now() - timestamp))
            
            # update worker parameters
            [p.put((actor.nn.get_weights(), critic.nn.get_weights())) for p in params_qs]
            
            timestamp = datetime.now()
            for b in range(epoch_size):
                shared_state_action_q.put((sample_weight[b], X[b:b + 1, :], U[b:b + 1, :]))
            print('state + actions distributed, took {}'.format(datetime.now() - timestamp))
                
            timestamp = datetime.now()
            policy_gradient *= 0
            for b in range(epoch_size):
                policy_gradient += shared_results_q.get() / epoch_size
            print('gradients calculated, additional time: {}'.format(datetime.now() - timestamp))

            actor.update_with_policy_gradient(policy_gradient)
            actor_target.soft_update(actor.nn.weights, lr=0.01)
            critic_target.soft_update(critic.nn.weights, lr=0.01)

            if datetime.now() > latest_plot + timedelta(seconds=15):
                r_avg, r_std = return_average(actor)
                r_avg_actor, r_std_actor = return_average(actor_target)
                print('beta: {} outer: {}/{} inner: {}/{} {}'.format(beta, a, n_iterations, i, n_inner, replay_buffer))
                print('return (mean/std): {:.3f} / {:.3f}, target: {:.3f} / {:.3f}'.format(r_avg, r_std, r_avg_actor, r_std_actor))
                latest_plot = datetime.now()

if __name__ == '__main__':
    try:
        fit()
    except KeyboardInterrupt:
        [p.terminate() for p in processes]
        exit(0)

[p.terminate() for p in processes]

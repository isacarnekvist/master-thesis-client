from __future__ import print_function

import sys
import json
import pickle
import threading
from operator import mul
from copy import deepcopy
from functools import reduce
from datetime import datetime, timedelta
from multiprocessing import Queue, Process, Value

if sys.version_info.major == 2:
    from Queue import Empty
else:
    from queue import Empty

import keras
import theano
import numpy as np
import theano.tensor as T
from keras import backend as K
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dense, Merge, Input, Lambda, merge, Layer, BatchNormalization


from ddpg import Actor, Critic
from naf.priority_buffer import PriorityBuffer
from environment import Environment, WIN, LOSE, NEUTRAL


theano.sandbox.cuda.use('cpu')


MAX_DIST = 0.01


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


def gradient_calculator(stop_flag, actor, critic, params_q, shared_state_action_q, shared_results_q):
    try:
        while not stop.value:
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


with open('actor_weights.json', 'r') as f:
    w = json.loads(f.read())
    actor.nn.set_weights(map(np.array, w))
    actor_target.nn.set_weights(map(np.array, w))
with open('critic_weights.json', 'r') as f:
    w = json.loads(f.read())
    critic.nn.set_weights(map(np.array, w))
    critic_target.nn.set_weights(map(np.array, w))


print('Starting gradient workers')
n_gradient_workers = 3
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


def training_loop():
    epoch_size = 1024
    batch_size = 32
    replay_buffer = PriorityBuffer(2 ** 20)
    gamma = 0.98
    epsilon = 0.1

    X = np.zeros((epoch_size, 6))
    Xp = np.zeros((epoch_size, 6))
    U = np.zeros((epoch_size, 2))
    R = np.zeros((epoch_size, 1))
    gradient_len = actor.gradients(X[:1, :]).shape[1]
    policy_gradient = np.zeros((1, gradient_len))

    n_iterations = 2048.0
    latest_plot = datetime.now() - timedelta(seconds=30)
    latest_trial_plot = datetime.now() - timedelta(seconds=60)
    a = 0
    for a in range(a, int(n_iterations)):
        print('iteration {} / {}'.format(a + 1, n_iterations))
        e.reset()
        latest_trial = []
        latest_rewards = []
        for b in range(epoch_size):
            x1 = e.get_state()
            mu = actor.predict(x1)
                
            noise = np.random.randn(1, 2) * MAX_DIST * 0.2
            mu = mu + noise
            dist = np.linalg.norm(mu)
            if dist > MAX_DIST:
                mu = mu * MAX_DIST / dist
            state, reward, x2 = e.interact(*(mu)[0, :])
            latest_trial.append(x2[0, :])
            latest_rewards.append(reward)
            replay_buffer.add({
                'x1': x1,
                'x2': x2,
                'u': mu,
                'r': reward
            }).set_value(10.0)
            if state in [LOSE, WIN] or b == epoch_size - 1 or len(latest_trial) > 128:
                latest_trial = []
                latest_rewards = []
                e.reset()
        
        n_inner = 16
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
            beta = np.exp((a - n_iterations) / (0.1 * n_iterations))
            sample_weight = np.array([1.0 / node.value for node in exp_nodes]) ** beta

            critic.nn.fit([X, U], Y, verbose=0, sample_weight=sample_weight, batch_size=batch_size, nb_epoch=16)
            
            for b in range(epoch_size):
                shared_state_action_q.put((sample_weight[b], X[b:b + 1, :], U[b:b + 1, :]))
                
            policy_gradient *= 0
            for b in range(epoch_size):
                policy_gradient += shared_results_q.get() / epoch_size

            actor.update_with_policy_gradient(policy_gradient, lr=0.1)
            actor_target.soft_update(actor.nn.weights, lr=0.001)
            critic_target.soft_update(critic.nn.weights, lr=0.001)
            
            if datetime.now() > latest_plot + timedelta(seconds=15):
                last_reward_avg, last_reward_std = last_reward_average(actor)
                print('last reward avg: {:.3f} std: {:.3f} beta: {:.3f} outer: {}/{} inner: {}/{} {}'.format(
                    last_reward_avg, last_reward_std, beta, a, n_iterations, i, n_inner, replay_buffer
                ))
                with open('actor_weights.json', 'w') as f:
                    w = list(map(lambda x: x.tolist(), actor_target.nn.get_weights()))
                    f.write(json.dumps(w))
                with open('critic_weights.json', 'w') as f:
                    w = list(map(lambda x: x.tolist(), critic_target.nn.get_weights()))
                    f.write(json.dumps(w))
                latest_plot = datetime.now()


if __name__ == '__main__':
    try:
        training_loop()
    except KeyboardInterrupt:
        print('stopping workers')
        stop.value = True

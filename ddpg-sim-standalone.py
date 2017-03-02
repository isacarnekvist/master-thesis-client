from __future__ import print_function
import json

import numpy as np
from theano import sandbox

from ddpg import Actor, Critic
from naf.priority_buffer import PriorityBuffer
from environment import Environment, WIN, LOSE, NEUTRAL

MAX_DIST = 0.01
sandbox.cuda.use('gpu0')
env = Environment(MAX_DIST, 'pushing-fixed-goal')

critic = Critic(env.get_state().shape[1], 2)
critic_target = Critic(env.get_state().shape[1], 2)
critic_target.clone_params(critic)

actor = Actor(env.get_state().shape[1], 2, critic, output_scaling=MAX_DIST)
actor_target = Actor(env.get_state().shape[1], 2, critic_target, output_scaling=MAX_DIST)
actor_target.clone_params(actor)


def return_average(env, actor, gamma=0.99):
    rewards = []
    for trial in range(32):
        np.random.seed(trial)
        env.reset()
        n_steps = 256
        return_ = 0.0
        for i in range(n_steps):
            mu = actor.predict(env.get_state())
            _, r, _ = env.interact(*mu.flatten())
            return_ += gamma ** i * r
        rewards.append(return_)
    return np.mean(rewards), np.std(rewards)


def sample_transition(env, priority_buffer, heuristic_prob=0.0):
    x1 = env.get_state()
    mu_rnd = MAX_DIST * np.tanh(75 * MAX_DIST * np.random.randn(2))
    mu_act = actor.predict(x1)[0, :]
    k = np.random.rand()
    mu = k * mu_rnd + (1 - k) * mu_act
    if np.random.rand() < heuristic_prob:
        mu = env.heuristic_move()
    end_state, reward, x2 = env.interact(*mu)
    priority_buffer.add({
        'x1': x1[0, :],
        'x2': x2[0, :],
        'reward': reward,
        'mu': mu,
        'end_state': end_state
    }).set_value(10.0)
    if end_state in [WIN, LOSE]:
        env.reset()


priority_buffer = PriorityBuffer(2 ** 20)

for i in range(512):
    sample_transition(env, priority_buffer, heuristic_prob=0.5)


def sample_batch(X, Xp, Y, U, R, S, critic, actor_target, critic_target, priority_buffer, gamma=0.99):
    nodes = []
    for i in range(X.shape[0]):
        sample = priority_buffer.sample()
        nodes.append(sample)
        X[i, :] = sample.data['x1']
        Xp[i, :] = sample.data['x2']
        U[i, :] = sample.data['mu']
        R[i, :] = sample.data['reward']
        S[i, :] = sample.data['end_state']
    Y[:, :] = R + gamma * critic_target.predict(Xp, actor_target.predict(Xp))
    [node.set_value(abs(e[0]) + 1e-6) for node, e in zip(nodes, critic.predict(X, U) - Y)]
    Y[S == WIN] = R[S == WIN]


n_iterations = 2500000
batch_size = 64

X = np.zeros((batch_size, env.get_state().shape[1]))
Xp = np.zeros((batch_size, env.get_state().shape[1]))
U = np.zeros((batch_size, 2))
Y = np.zeros((batch_size, 1))
R = np.zeros((batch_size, 1))
S = np.zeros((batch_size, 1))

returns = []

iteration = 0
for iteration in range(iteration, n_iterations):
    sample_transition(env, priority_buffer)
    sample_batch(X, Xp, Y, U, R, S, critic, actor_target, critic_target, priority_buffer)
    critic.fit(X, U, Y)
    actor.fit(X)
    critic_target.soft_update(critic)
    actor_target.soft_update(actor)
    if iteration % 64 == 0:
        r_avg, r_std = return_average(env, actor_target)
        print('Return average:', r_avg)
        returns.append(r_avg)
        with open('ddpg-returns.txt', 'w') as f:
            f.write(json.dumps(returns))
        actor.save_params('actor_params.txt')
        actor_target.save_params('actor_target_params.txt')
        critic.save_params('critic_params.txt')
        critic_target.save_params('critic_target_params.txt')

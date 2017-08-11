import theano
import theano.sandbox
import theano.sandbox.cuda


import numpy as np

from a3c import Critic, Actor
from environment import Environment

GAMMA = 0.99
MAX_DIST = 0.01
mode = 'pushing-moving-goal'
theano.sandbox.cuda.use('gpu0')


def rewards2R(rewards, gamma=0.99):
    R = 0.0
    Rs = []
    for r in rewards[::-1]:
        R = r + gamma * R
        Rs.append(R)
    return list(reversed(Rs))[:-1]


env = Environment(MAX_DIST, mode)
critic = Critic(env.get_state().shape[0], adam_beta1=0.9, hidden_size=200)
critic.load_params('./critic_supervised.txt')
actor = Actor(env.get_state().shape[0], adam_beta1=0.9, hidden_size=200)


def run_steps(env, critic, t_max=5):
    X = []
    U = []
    R = []
    V = []
    x = env.get_state()
    X.append(x)
    done = False
    rewards = []
    while not done:
        u = env.heuristic_move()
        if np.random.rand() < 0.25:
            theta = 2 * np.pi * np.random.rand()
            d = np.random.rand()
            u = d * np.array([np.cos(theta), np.sin(theta)])
        U.append(u)
        V.append(critic.predict(np.array([X[-1]]))[0])
        x, reward, done, state = env.step(u)
        X.append(x)
        rewards.append(reward)
        if len(X) == t_max + 1:
            break
    if done:
        env.reset()
    rewards.append(critic.predict(np.array([X[-1]]))[0, 0])
    return np.array(X[:-1]), np.array(U), np.array([rewards2R(rewards)]).T, np.array(V)


while True:
    n_average_steps = 1024
    loss = 0.0
    for _ in range(n_average_steps):
        for _ in range(32):
            x, u, r, v = run_steps(env, critic)
            actor.update(actor.gradients(x, u, r, v))
        loss += actor.loss(x, u, r, v) / n_average_steps
    print(loss)
    actor.save_params('actor_supervised.txt')

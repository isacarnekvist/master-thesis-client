import numpy as np

from a3c import Critic
from environment import Environment

GAMMA = 0.99
MAX_DIST = 0.01
mode = 'pushing-moving-goal'


def rewards2R(rewards, gamma=0.99):
    R = 0.0
    Rs = []
    for r in rewards[::-1]:
        R = r + gamma * R
        Rs.append(R)
    return list(reversed(Rs))[:-1]


env = Environment(MAX_DIST, mode)
critic = Critic(env.get_state().shape[0], adam_beta1=0.9, hidden_size=200)


def episode(env):
    X = []
    R = []
    x = env.reset()
    X.append(x)
    done = False
    rewards = []
    while not done:
        x, reward, done, _ = env.step(env.heuristic_move())
        X.append(x)
        rewards.append(reward)
    rewards.append(0.0)
    R = rewards2R(rewards)
    R.append(0.0)
    return np.array(X), np.array([R]).T


n_iterations = 4096
for i in range(4096):
    losses = 0.0
    for _ in range(64):
        for _ in range(64):
            X, R = episode(env)
            critic.update(critic.gradients(X, R))
        losses += critic.loss(X, R) / 64
    print('iteration {}/{}, mse: {}'.format(i + 1, n_iterations, losses))
    critic.save_params('critic_supervised.txt')

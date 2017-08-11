import json
import multiprocessing
from time import sleep
from queue import Full
from datetime import datetime
from multiprocessing import Process, Queue


GAMMA = 0.99
MAX_DIST = 0.01
mode = 'pushing-fixed-cube'


def rewards2R(rewards, gamma=0.99):
    R = 0.0
    Rs = []
    for r in rewards[::-1]:
        R = r + gamma * R
        Rs.append(R)
    return list(reversed(Rs))[:-1]


def clip_gradient(gradient):
    norm = np.linalg.norm(gradient)
    if norm > 40.0:
        return 40.0 * gradient / norm
    else:
        return gradient


def trainer(actor_params, critic_params, actor_gradients, critic_gradients):
    import numpy as np
    from a3c import Actor, Critic
    from environment import Environment, WIN, LOSE, NEUTRAL
    
    np.random.seed(datetime.now().microsecond)
    
    episode_length = 0
    env = Environment(MAX_DIST, mode)
    actor = Actor(env.get_state().shape[1])
    critic = Critic(env.get_state().shape[1])
    actor.load_params('actor_params.txt')
    critic.load_params('critic_params.txt')
    
    while True:
        if episode_length > 32:
            env.reset()
            episode_length = 0
        episode_length += 1
        if not actor_params.empty():
            actor.set_params(actor_params.get())
        if not critic_params.empty():
            critic.set_params(critic_params.get())
        
        X = []
        U = []
        x2s = []
        rewards = []
        for i in range(5):
            mu, sigma = actor.predict(env.get_state())
            dx, dy = np.random.multivariate_normal(mu[0, :], np.diag(sigma[0, :]))
            X.append(env.get_state().flatten())
            U.append([dx, dy])
            state, reward, x2 = env.interact(dx, dy)
            x2s.append(x2.flatten())
            rewards.append(reward)
            if state in [WIN, LOSE]:
                env.reset()
                break
        V = critic.predict(X)
        if state == NEUTRAL:
            rewards.append(V[-1, 0])
        elif state == WIN:
            rewards.append(0.0)
        else:
            rewards.append(reward * 100)
        R = rewards2R(rewards, gamma=GAMMA)
        X = np.array(X)
        U = np.array(U)
        R = np.array([R]).T
        ga = list(map(clip_gradient, actor.gradients(X, U, R, V)))
        gc = list(map(clip_gradient, critic.gradients(X, R)))
        actor_gradients.put(ga)
        critic_gradients.put(gc)
        
n_processes = 6
actor_gradient_queue = Queue(maxsize=n_processes)
critic_gradient_queue = Queue(maxsize=n_processes)
pool = []
actor_param_queues = []
critic_param_queues = []
for _ in range(n_processes):
    apq = Queue(maxsize=1)
    cpq = Queue(maxsize=1)
    actor_param_queues.append(apq)
    critic_param_queues.append(cpq)
    pool.append(
        Process(
            target=trainer,
            args=(
                apq, cpq, actor_gradient_queue, critic_gradient_queue
            )
        )
    )

[p.start() for p in pool]
sleep(15.0)


import numpy as np

from a3c import Actor, Critic
from environment import Environment, WIN, LOSE, NEUTRAL


def return_average(env, actor, gamma=0.99):
    rewards = []
    for trial in range(32):
        np.random.seed(trial)
        env.reset()
        n_steps = 256
        return_ = 0.0
        for i in range(n_steps):
            mu, _ = actor.predict(env.get_state())
            _, r, _ = env.interact(*mu.flatten())
            return_ += gamma ** i * r
        rewards.append(return_)
    return np.mean(rewards), np.std(rewards)


try:
    with open('averages.txt', 'w') as f:
        averages = json.loads(f.read())
except IOError:
    print('Could not load previous averages')
    averages = []

env = Environment(MAX_DIST, mode)
actor = Actor(env.get_state().shape[1], adam_beta1=0.0)
critic = Critic(env.get_state().shape[1], adam_beta1=0.0)
try:
    actor.load_params('actor_params.txt')
    critic.load_params('critic_params.txt')
except IOError:
    print('Could not load previous parameters')


best_score = -np.inf
r_avg, r_std = return_average(env, actor)
averages.append(r_avg)
if r_avg > best_score:
    best_score = r_avg
i = len(averages) * 1024
for i in range(i, 2 ** 20):
    cg = critic_gradient_queue.get(timeout=1)
    critic.update(cg)
    for c in critic_param_queues:
        try:
            c.put(critic.params, block=False)
        except Full:
            pass
    ag = actor_gradient_queue.get(timeout=1)
    actor.update(ag)
    for a in actor_param_queues:
        try:
            a.put(actor.params, block=False)
        except Full:
            pass
    if i % 1024 == 0:
        r_avg, r_std = return_average(env, actor)
        averages.append(r_avg)
        if r_avg > best_score:
            with open('averages.txt', 'w') as f:
                f.write(json.dumps(averages))
            actor.save_params('actor_params.txt')
            critic.save_params('critic_params.txt')
            best_score = r_avg

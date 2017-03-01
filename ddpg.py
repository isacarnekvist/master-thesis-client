"""
Theano implementation of Deep Deterministic Policy Gradient (DDPG) [1]

[1] Lillicrap, Timothy P., et al. "Continuous control with deep reinforcement
learning." arXiv preprint arXiv:1509.02971 (2015).
"""

import json
import theano
import numpy as np
import theano.tensor as T


class Adam:
    
    def __init__(self, lr=0.001):
        self.alpha = lr
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0
        
    def get_update(self, gradient):
        """
        Return the update to be added to model parameters (for minimizing)
        """
        if self.t == 0:
            self.m = gradient * 0
            self.v = gradient * 0
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradient ** 2
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        return -self.alpha * m_hat / (np.sqrt(v_hat) + self.epsilon)


class NN:
    
    def soft_update(self, nn_other, tau=0.001):
        for p_self, p_other in zip(self.params, nn_other.params):
            p_self.set_value(tau * p_other.get_value() + (1 - tau) * p_self.get_value())

    def clone_params(self, nn_other):
        self.soft_update(nn_other, tau=1.0)

    def load_params(self, filename):
        with open(filename, 'r') as f:
            params_list = json.loads(f.read())
        [p.set_value(p_saved) for p, p_saved in zip(self.params, params_list)]

    def save_params(self, filename):
        params_list = [p.get_value().tolist() for p in self.params]
        with open(filename, 'w') as f:
            f.write(json.dumps(params_list))


class Critic(NN):
    
    def construct_q(self, x, u):
        a1 = T.dot(x, self.fc1_w) + self.fc1_b
        y1 = T.horizontal_stack(T.nnet.relu(a1), u)
        a2 = T.dot(y1, self.fc2_w) + self.fc2_b
        y2 = T.nnet.relu(a2)
        return T.dot(y2, self.fc3_w) + self.fc3_b

    def __init__(self, x_size, u_size, hidden_sizes=(400, 300), lr=1e-3):
        """Critic network in Deep Deterministic Policy Gradient (DDPG)
        x_size : int
            state space dimensionality
        u_size : int
            action space dimensionality
        hidden_sizes : (int, int), optional
            sizes of the two hidden layers, default is (400, 300)
        lr : float, optional
            default is 1e-3
        """
        # Inputs
        x = T.fmatrix('State')
        u = T.fmatrix('Controls')

        # Shared parameters
        self.fc1_w = theano.shared(
            2 / np.sqrt(x_size) * (np.random.rand(x_size, hidden_sizes[0]) - 0.5)
        )
        self.fc1_b = theano.shared(
            np.zeros(hidden_sizes[0])
        )
        self.fc2_w = theano.shared(
            2 / np.sqrt(hidden_sizes[0]) * (np.random.rand(hidden_sizes[0] + u_size, hidden_sizes[1]) - 0.5)
        )
        self.fc2_b = theano.shared(
            np.zeros(hidden_sizes[1])
        )
        self.fc3_w = theano.shared(
            6 * 1e-3 * (np.random.rand(hidden_sizes[1], 1) - 0.5)
        )
        self.fc3_b = theano.shared(
            np.zeros(1)
        )
        params = [self.fc1_w, self.fc1_b, self.fc2_w, self.fc2_b, self.fc3_w, self.fc3_b]
        self.params = params
        self.adams = [Adam(lr=lr) for _ in params]
        
        self.q = self.construct_q(x, u)
        
        y = T.fmatrix('Targets')
        weight_decay = 1e-2 * T.sum([(p ** 2).sum() for p in params])
        loss = 1.0 / y.shape[0] * ((y - self.q) ** 2).sum() + weight_decay
        gradients = T.grad(loss, wrt=params)
        self.gradients = theano.function([x, u, y], gradients, allow_input_downcast=True)
        
        self.predict = theano.function([x, u], self.q, allow_input_downcast=True)
        
    def fit(self, X, U, Y):
        for adam, param, gradient in zip(self.adams, self.params, self.gradients(X, U, Y)):
            param.set_value(param.get_value() + adam.get_update(gradient))


class Actor(NN):
    
    def __init__(self, x_size, u_size, critic, hidden_sizes=(400, 300), output_scaling=1.00, lr=1e-4):
        """Actor network in Deep Deterministic Policy Gradient (DDPG)
        """
        x = T.fmatrix('State')

        fc1_w = theano.shared(
            2 / np.sqrt(x_size) * (np.random.rand(x_size, hidden_sizes[0]) - 0.5)
        )
        fc1_b = theano.shared(
            np.zeros(hidden_sizes[0])
        )

        a1 = T.dot(x, fc1_w) + fc1_b
        y1 = T.nnet.relu(a1)

        fc2_w = theano.shared(
            2 / np.sqrt(hidden_sizes[0]) * (np.random.rand(hidden_sizes[0], hidden_sizes[1]) - 0.5)
        )
        fc2_b = theano.shared(
            np.zeros(hidden_sizes[1])
        )
        a2 = T.dot(y1, fc2_w) + fc2_b
        y2 = T.nnet.relu(a2)

        fc3_w = theano.shared(
            6 * 1e-3 * (np.random.rand(hidden_sizes[1], u_size) - 0.5)
        )
        fc3_b = theano.shared(
            np.zeros(u_size)
        )
        a3 = T.dot(y2, fc3_w) + fc3_b
        u = output_scaling * T.tanh(a3)

        params = [fc1_w, fc1_b, fc2_w, fc2_b, fc3_w, fc3_b]
        self.params = params
        self.adams = [Adam(lr=lr) for _ in params]
        
        self.q = critic.construct_q(x, u)
        
        sample_weights = T.fmatrix('sample_weights')
        loss = -(self.q * sample_weights).sum()
        gradients = T.grad(loss, wrt=params)
        
        self.gradients = theano.function([x, sample_weights], gradients, allow_input_downcast=True)
        self.predict = theano.function([x], u, allow_input_downcast=True)
        
    def fit(self, X, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones((X.shape[0], 1))
        else:
            sample_weight = np.array([sample_weight]).T
        for param, grad, adam in zip(self.params, self.gradients(X, sample_weight), self.adams):
            update = adam.get_update(grad)
            param.set_value(param.get_value() + update)

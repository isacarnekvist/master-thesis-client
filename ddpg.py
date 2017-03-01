"""
Theano implementation of Deep Deterministic Policy Gradient (DDPG) [1]

[1] Lillicrap, Timothy P., et al. "Continuous control with deep reinforcement
learning." arXiv preprint arXiv:1509.02971 (2015).
"""

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


class Critic(NN):
    
    def __init__(self, x_size, u_size, batch_size=64, hidden_sizes=(400, 300), lr=1e-3):
        """Critic network in Deep Deterministic Policy Gradient (DDPG)
        x_size : int
            state space dimensionality
        u_size : int
            action space dimensionality
        batch_size : int, optional
            default is 64
        hidden_sizes : (int, int), optional
            sizes of the two hidden layers, default is (400, 300)
        lr : float, optional
            default is 1e-3
        """
        self.batch_size = batch_size

        x = T.fmatrix('State')
        self.x = x

        fc1_w = theano.shared(
            2 / np.sqrt(x_size) * (np.random.rand(x_size, hidden_sizes[0]) - 0.5)
        )
        fc1_b = theano.shared(
            np.zeros(hidden_sizes[0])
        )
        
        u = T.fmatrix('Controls')

        a1 = T.dot(x, fc1_w) + fc1_b
        y1 = T.horizontal_stack(T.nnet.relu(a1), u)

        fc2_w = theano.shared(
            2 / np.sqrt(hidden_sizes[0]) * (np.random.rand(hidden_sizes[0] + u_size, hidden_sizes[1]) - 0.5)
        )
        fc2_b = theano.shared(
            np.zeros(hidden_sizes[1])
        )
        a2 = T.dot(y1, fc2_w) + fc2_b
        y2 = T.nnet.relu(a2)

        fc3_w = theano.shared(
            6 * 1e-3 * (np.random.rand(hidden_sizes[1], 1) - 0.5)
        )
        fc3_b = theano.shared(
            np.zeros(1)
        )
        y3 = T.dot(y2, fc3_w) + fc3_b

        params = [fc1_w, fc1_b, fc2_w, fc2_b, fc3_w, fc3_b]
        self.params = params
        self.adams = [Adam(lr=lr) for _ in params]
        
        y = T.fmatrix('Targets')
        loss = 1.0 / y.shape[0] * ((y - y3) ** 2).sum()
        gradients = T.grad(loss, wrt=params)
        self.gradients = theano.function([x, u, y], gradients, allow_input_downcast=True)
        
        gradients_du = T.zeros((batch_size, 1, u_size))
        for sample in range(batch_size):
            gradients_du = T.set_subtensor(
                gradients_du[sample, 0, :],
                T.grad(y3[sample, 0], wrt=u)[sample, :]
            )
        self.gradients_du = theano.function([x, u], gradients_du, allow_input_downcast=True)
        self.predict = theano.function([x, u], y3, allow_input_downcast=True)
        
    def fit(self, X, U, Y):
        for adam, param, gradient in zip(self.adams, self.params, self.gradients(X, U, Y)):
            param.set_value(param.get_value() + adam.get_update(gradient))


class Actor(NN):
    
    def __init__(self, x_size, u_size, batch_size=64, hidden_sizes=(400, 300), output_scaling=1.00, lr=1e-4):
        """Actor network in Deep Deterministic Policy Gradient (DDPG)
        """
        self.batch_size = batch_size
        x = T.fmatrix('State')
        self.x = x

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
        y3 = output_scaling * T.tanh(a3)

        params = [fc1_w, fc1_b, fc2_w, fc2_b, fc3_w, fc3_b]
        self.params = params
        self.adam = Adam(lr=lr)
        self.n_params_all = [p.flatten().shape.eval()[0] for p in params]
        self.params_shapes = [p.shape.eval() for p in params]
        n_params = sum(self.n_params_all)
        gradients = T.zeros((batch_size, u_size, n_params))
        for sample in range(batch_size):
            for dim in range(u_size):
                grads = []
                for i, grad in enumerate(T.grad(y3[sample, dim], wrt=params)):
                    grads.append(grad.reshape((1, self.n_params_all[i])))
                gradients = T.set_subtensor(
                    gradients[sample, dim, :],
                    T.horizontal_stack(*grads).flatten()
                )
        self.gradients = theano.function([x], gradients, allow_input_downcast=True)
        self.predict = theano.function([x], y3, allow_input_downcast=True)
        
    def fit(self, X, U, critic, sample_weight=None):
        dc = critic.gradients_du(X, U)
        da = self.gradients(X)
        if sample_weight is not None:
            grad_all = np.matmul(dc, da)
            grad = self.adam.get_update(grad_all.sum(axis=0))
        else:
            grad = self.adam.get_update(np.matmul(dc, da).sum(axis=0))
        ind = 0
        for n_params, shape, param in zip(self.n_params_all, self.params_shapes, self.params):
            update = grad[0, ind:ind + n_params].reshape(shape)
            param.set_value(param.get_value() - update)
            ind += n_params

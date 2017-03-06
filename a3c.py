import json

import theano
import numpy as np
import theano.tensor as T


class Adam:
    
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.alpha = lr
        self.beta1 = beta1
        self.beta2 = beta2
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


class A3C:

    def set_params(self, params):
        for param, param_new in zip(self.params, params):
            param.set_value(param_new.get_value())

    def update(self, gradients):
        for param, grad, adam in zip(self.params, gradients, self.adams):
            param.set_value(param.get_value() + adam.get_update(grad))

    def load_params(self, filename):
        with open(filename, 'r') as f:
            params_list = json.loads(f.read())
        [p.set_value(p_saved) for p, p_saved in zip(self.params, params_list)]

    def save_params(self, filename):
        params_list = [p.get_value().tolist() for p in self.params]
        with open(filename, 'w') as f:
            f.write(json.dumps(params_list))


class Actor(A3C):
    
    def __init__(self, x_size, u_size=2, hidden_size=200, mu_scaling=0.01, adam_beta1=0.0, adam_beta2=0.999):
        x = T.fmatrix('state')
        fc1_w = theano.shared(
            2 / np.sqrt(x_size) * (np.random.rand(x_size, hidden_size) - 0.5)
        )
        fc1_b = theano.shared(
            np.zeros(hidden_size)
        )
        y1 = T.nnet.relu(x.dot(fc1_w) + fc1_b)

        fc2_w = theano.shared(
            2 / np.sqrt(hidden_size) * (np.random.rand(hidden_size, hidden_size) - 0.5)
        )
        fc2_b = theano.shared(
            np.zeros(hidden_size)
        )
        y2 = T.nnet.relu(y1.dot(fc2_w) + fc2_b)
        
        mu_w = theano.shared(
            6 * 1e-3 * (np.random.rand(hidden_size, u_size) - 0.5)
        )
        mu_b = theano.shared(
            np.zeros(u_size)
        )
        mu = mu_scaling * T.tanh(y2.dot(mu_w) + mu_b)
        
        sigma_w = theano.shared(
            (np.random.rand(hidden_size, u_size) - 0.5)
        )
        sigma_b = theano.shared(
            -5.0 * np.ones(u_size)
        )
        sigma = mu_scaling * T.nnet.softplus((y1.dot(sigma_w) + sigma_b))
        self.predict = theano.function([x], [mu, sigma], allow_input_downcast=True)
        
        u = T.fmatrix('actions')
        det = sigma[:, 0:1] * sigma[:, 1:] # sigma.prod(axis=1, keepdims=True) does not work
        log_probability_of_u = (
            -0.5 * (
                T.log(sigma).sum(axis=1, keepdims=True) +
                ((mu - u) ** 2 / sigma).sum(axis=1, keepdims=True)
            )
        )
        self.params = [fc1_w, fc1_b, fc2_w, fc2_b, mu_w, mu_b, sigma_w, sigma_b]
        self.adams = [Adam(lr=1e-4, beta1=adam_beta1, beta2=adam_beta2) for _ in self.params]

        v = T.fmatrix('value_targets')
        r = T.fmatrix('return_targets')
        beta = 1e-4
        loss = (-log_probability_of_u * (r - v) - beta * T.log(det)).sum()
        self.loss = theano.function([x, u, r, v], loss, allow_input_downcast=True)
        self.gradients = theano.function(
            [x, u, r, v],
            T.grad(loss, wrt=self.params),
            allow_input_downcast=True
        )

        
class Critic(A3C):
    
    def __init__(self, x_size, u_size=2, hidden_size=100, adam_beta1=0.0, adam_beta2=0.999):
        x = T.fmatrix('state')
        
        fc1_w = theano.shared(
            2 / np.sqrt(x_size) * (np.random.rand(x_size, hidden_size) - 0.5)
        )
        fc1_b = theano.shared(
            np.zeros(hidden_size)
        )
        y1 = T.nnet.relu(x.dot(fc1_w) + fc1_b)

        fc2_w = theano.shared(
            2 / np.sqrt(hidden_size) * (np.random.rand(hidden_size, hidden_size) - 0.5)
        )
        fc2_b = theano.shared(
            np.zeros(hidden_size)
        )
        y2 = T.nnet.relu(y1.dot(fc2_w) + fc2_b)
        
        v_w = theano.shared(
            2 / np.sqrt(hidden_size) * (np.random.rand(hidden_size, 1) - 0.5)
        )
        v_b = theano.shared(
            np.zeros(1)
        )
        v = y2.dot(v_w) + v_b
        self.predict = theano.function([x], v, allow_input_downcast=True)
        
        self.params = [fc1_w, fc1_b, fc2_w, fc2_b, v_w, v_b]
        self.adams = [Adam(lr=1e-4, beta1=adam_beta1, beta2=adam_beta2) for _ in self.params]
        
        r = T.fmatrix('return_targets')
        loss = ((r - v) ** 2).sum()
        gradients = T.grad(loss, wrt=self.params)
        self.gradients = theano.function([x, r], gradients, allow_input_downcast=True)

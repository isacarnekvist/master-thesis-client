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
        self.adams = [Adam(lr=1e-5, beta1=adam_beta1, beta2=adam_beta2) for _ in self.params]

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


class ActorMM(A3C):
    
    def __init__(self, x_size, n_modes, u_size=2, hidden_size=200, mu_scaling=0.01, adam_beta1=0.0, adam_beta2=0.999):
        self.n_modes = n_modes
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
        
        # mixture_weights
        mw_w = theano.shared(
            2 / np.sqrt(hidden_size) * (np.random.rand(hidden_size, n_modes) - 0.5)
        )
        mw_b = theano.shared(
            np.zeros(n_modes)
        )
        mw_y = T.nnet.nnet.softmax(y2.dot(mw_w) + mw_b)
        
        self.mw = theano.function([x], mw_y, allow_input_downcast=True)
        
        self.params = []
        mus = []
        sigmas = []
        for n in range(n_modes):
            mu_w = theano.shared(
                6 * 1e-3 * (np.random.rand(hidden_size, u_size) - 0.5)
            )
            mu_b = theano.shared(
                np.zeros(u_size)
            )
            mu = mu_scaling * T.tanh(y2.dot(mu_w) + mu_b)
            mus.append(mu)
        
            sigma_w = theano.shared(
                (np.random.rand(hidden_size, u_size) - 0.5)
            )
            sigma_b = theano.shared(
                -5.0 * np.ones(u_size)
            )
            sigma = mu_scaling ** 2 * T.nnet.sigmoid((y1.dot(sigma_w) + sigma_b))
            sigmas.append(sigma)
            self.params.extend([mu_w, mu_b, sigma_w, sigma_b])

        self._predict = theano.function([x], [mw_y] + mus + sigmas, allow_input_downcast=True)

        u = T.fmatrix('actions')
        mixture_probs = T.zeros((u.shape[0], n_modes))
        for n in range(n_modes):
            d = u - mus[n]
            log_det = -0.5 * T.log(sigmas[n]).sum(axis=1)
            exp_arg = -0.5 * (d * d / sigmas[n]).sum(axis=1)
            mixture_probs = T.set_subtensor(
                mixture_probs[:, n],
                T.log(mw_y[:, n]) - np.log(2 * np.pi) + log_det + exp_arg
            )
        tot_log_prob = T.log(T.exp(mixture_probs).sum(axis=1, keepdims=True))
        self.mp = theano.function([x, u], tot_log_prob, allow_input_downcast=True)
        
        # TODO add all params
        # add loss for mu similarity?
        self.params.extend([fc1_w, fc1_b, fc2_w, fc2_b, mu_w, mu_b, sigma_w, sigma_b])
        self.adams = [Adam(lr=1e-5, beta1=adam_beta1, beta2=adam_beta2) for _ in self.params]

        v = T.fmatrix('value_targets')
        r = T.fmatrix('return_targets')
        beta = 1e-4
        #loss = (-log_probability_of_u * (r - v) - beta * T.log(det)).sum()
        loss = (-tot_log_prob * (r - v) - T.log((mus[0] - mus[1]) ** 2) - T.log(mw_y)).sum()
        self.loss = theano.function([x, u, r, v], loss, allow_input_downcast=True)
        self.gradients = theano.function(
            [x, u, r, v],
            T.grad(loss, wrt=self.params),
            allow_input_downcast=True
        )

    def predict(self, x):
        p = self._predict(x)
        mw = p[0][:, 0]
        mode = np.argmax(np.random.multinomial(1, mw))
        mu = p[1 + mode][0, :]
        sigma = np.diag(p[1 + self.n_modes + mode][0, :])
        return np.random.multivariate_normal(mean=mu, cov=sigma)


class ActorFullCov(A3C):
    
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
        
        l_diag_w = theano.shared(
            (np.random.rand(hidden_size, u_size) - 0.5)
        )
        l_diag_b = theano.shared(
            np.zeros(u_size)
        )
        l_diag = mu_scaling * T.nnet.softplus((y1.dot(l_diag_w) + l_diag_b))
        cov_w = theano.shared(
            (np.random.rand(hidden_size, 1) - 0.5)
        )
        cov_b = theano.shared(
            np.zeros(1)
        )
        cov = mu_scaling * (y1.dot(cov_w) + cov_b)
        
        if u_size != 2:
            raise NotImplementedError('u_size must be 2')
        L = T.zeros((x.shape[0], 2, 2))
        L = T.set_subtensor(L[:, [0, 1], [0, 1]], l_diag)
        L = T.set_subtensor(L[:, 1, 0], cov.flatten())
        sigma = T.batched_dot(L, L.swapaxes(1, 2))
        det = (sigma[:, 0, 0] * sigma[:, 1, 1] - sigma[:, 0, 1] * sigma[:, 1, 0]).reshape((x.shape[0], 1, 1))
        swap_diag = T.set_subtensor(sigma[:, 0, 0], sigma[:, 1, 1])
        swap_diag = T.set_subtensor(swap_diag[:, 1, 1], sigma[:, 0, 0])
        sigma_inv = 1 / det * T.set_subtensor(swap_diag[:, [0, 1], [1, 0]], -swap_diag[:, [0, 1], [1, 0]])
        entropy = T.log(det)
        
        self.det = theano.function([x], det, allow_input_downcast=True)
        self.sigma_inv = theano.function([x], sigma_inv, allow_input_downcast=True)
        self.predict = theano.function([x], [mu, sigma], allow_input_downcast=True)
        
        u = T.fmatrix('actions')
        d = (u - mu).reshape((x.shape[0], 1, 2))
        log_probability_of_u = -0.5 * (
            2 * T.log(2 * np.pi) +
            T.log(det) +
            T.batched_dot(d, T.batched_dot(sigma_inv, d.swapaxes(1, 2)))
        ).reshape((x.shape[0], 1))
        self.params = [fc1_w, fc1_b, fc2_w, fc2_b, mu_w, mu_b, l_diag_w, l_diag_b, cov_w, cov_b]
        self.adams = [Adam(lr=1e-5, beta1=adam_beta1, beta2=adam_beta2) for _ in self.params]

        v = T.fmatrix('value_targets')
        r = T.fmatrix('return_targets')
        beta = 1e-4
        loss = (-log_probability_of_u * (r - v) - beta * T.log(det)).sum()
        self.loss = theano.function([x, u, r, v], loss, allow_input_downcast=True)
        self.log_probability_of_u = theano.function([x, u], log_probability_of_u, allow_input_downcast=True, on_unused_input='ignore')
        self.gradients = theano.function(
            [x, u, r, v],
            T.grad(loss, wrt=self.params),
            allow_input_downcast=True
        )


class Critic(A3C):
    
    def __init__(self, x_size, u_size=2, hidden_size=200, adam_beta1=0.0, adam_beta2=0.999):
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
        self.adams = [Adam(lr=1e-5, beta1=adam_beta1, beta2=adam_beta2) for _ in self.params]
        
        r = T.fmatrix('return_targets')
        loss = ((r - v) ** 2).sum() / r.shape[0]
        self.loss = theano.function([x, r], loss, allow_input_downcast=True)
        gradients = T.grad(loss, wrt=self.params)
        self.gradients = theano.function([x, r], gradients, allow_input_downcast=True)

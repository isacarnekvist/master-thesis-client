from operator import mul
from functools import reduce

import keras
import theano
import numpy as np
import theano.tensor as T
from keras import backend as K
from keras.optimizers import Adam
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
        
        self.nn = Sequential([
            Merge([x_model, u_model], mode='concat'),
            BatchNormalization(input_shape=(x_size,), name='bn1'),
            Dense(output_dim=hidden_size, activation='relu', name='fc1'),
            BatchNormalization(name='bn2'),
            Dense(output_dim=hidden_size, activation='relu', name='fc2'),
            BatchNormalization(name='bn3'),
            Dense(output_dim=(1), name='Q'),
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
        x = Input(shape=(x_size, ), name='state')
        self.nn = Sequential([
            BatchNormalization(input_shape=(x_size,), name='bn1'),
            Dense(input_shape=(x_size,), output_dim=hidden_size, activation='relu', name='fc1'),
            BatchNormalization(name='bn2'),
            Dense(output_dim=hidden_size, activation='relu', name='fc2'),
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
    
    def update_with_policy_gradient(self, policy_gradient, lr=0.0001):
        """
        Update from separate actor and critic gradients, which
        multiply to make the policy gradient
        """
        i = 0
        policy_gradient = policy_gradient.astype(np.float32)
        for g in self.trainable_weights:
            v = g.get_value()
            param_len = reduce(mul, v.shape)
            g.set_value(v + lr * policy_gradient[0, i:i + param_len].reshape(v.shape))
            i += param_len

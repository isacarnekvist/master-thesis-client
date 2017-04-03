import json

import keras
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization, Input, merge, Merge


class DDPG:

    def load_params(self, filename):
        with open(filename, 'r') as f:
            params_list = json.loads(f.read())
        [p.set_value(p_saved) for p, p_saved in zip(self.q.weights, params_list)]

    def save_params(self, filename):
        params_list = [p.get_value().tolist() for p in self.q.weights]
        with open(filename, 'w') as f:
            f.write(json.dumps(params_list))


class Critic(DDPG):
    
    def build(self, x, u):
        y = merge([x, u], mode='concat', name='x_u_merge')
        for layer in self.layers:
            y = layer(y)
        return y
    
    def __init__(self, x_size, u_size):
        x = Input(shape=(x_size,))
        u = Input(shape=(u_size,))
        self.layers = []
        self.layers = [
            BatchNormalization(name='critic_bn1', input_shape=(x_size + u_size,)),
            Dense(input_dim=(x_size,), W_regularizer=l2(), activation='elu', output_dim=400, name='critic_fc1'),
            BatchNormalization(name='critic_bn2'),
            Dense(output_dim=300, W_regularizer=l2(), activation='elu', name='critic_fc2'),
            Dense(output_dim=1, W_regularizer=l2(), name='critic_q'),
        ]
        self.q = Model(input=[x, u], output=self.build(x, u))
        self.q.compile(loss='mse', optimizer=Adam(1e-3))
    
    def soft_update(self, other, tau=0.001):
        for a, b in zip(self.q.weights, other.q.weights):
            a.set_value((1 - tau) * a.get_value() + tau * b.get_value())


def loss(y_true, y_pred):
    return -y_pred


class Actor(DDPG):
    
    def __init__(self, x_size, u_size, critic):
        x = Input(shape=(x_size,), name='actor_x')
        self.u = Sequential([
            BatchNormalization(input_shape=(x_size,), name='actor_bn1'),
            Dense(output_dim=400, name='actor_fc1', activation='elu'),
            BatchNormalization(name='actor_bn2'),
            Dense(output_dim=300, name='actor_fc2', activation='elu'),
            Dense(output_dim=u_size, name='actor_fc3', activation='tanh')
        ], name='actor_u_model')
        q = critic.build(x, self.u(x))
        self.q = Model(input=x, output=q, name='actor_q_part_model')
        layers = []
        for l in self.q.layers:
            if hasattr(l, 'trainable') and l.trainable:
                if l.name.startswith('critic'):
                    layers.append(l)
                    l.trainable = False
        self.q.compile(loss=loss, optimizer=Adam(1e-4))
        for l in layers:
            l.trainable = True # reset

    def soft_update(self, other, tau=0.001):
        for a, b in zip(self.u.weights, other.u.weights):
            a.set_value((1 - tau) * a.get_value() + tau * b.get_value())

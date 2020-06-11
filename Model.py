import keras
from keras.layers import Dense
from keras.layers import Add
from keras.layers import Subtract
from keras.layers import Lambda
from keras.layers.advanced_activations import LeakyReLU
import keras.backend as K
import numpy as np

class Model:
    def __init__(self, num_states, num_actions, batch_size=0, dueling=True):
        self.num_states = num_states
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.dueling = dueling
        
        # placeholders - will be initialized in define_model
        self.nn = None
        
        self.define_model()
    
    def define_model(self):
        ALPHA = 0.6
        inputs = keras.Input(shape=(self.num_states,))
        x = Dense(100)(inputs)
        x = LeakyReLU(alpha=ALPHA)(x)
        if self.dueling:
            state_value = Dense(1)(x)
            state_value = LeakyReLU(alpha=ALPHA)(state_value)
            state_value = Lambda(lambda s: K.expand_dims(s[:, 0], -1), output_shape=(self.num_actions,))(state_value)
            
            advantage = Dense(self.num_actions)(x)
            advantage = LeakyReLU(alpha=ALPHA)(advantage)
            advantage = Lambda(lambda a: a[...] - K.mean(a[...], keepdims=True), output_shape=(self.num_actions,))(advantage)
            
            x = Add()([state_value, advantage])
        else:
            x = Dense(self.num_actions)(x)
            x = LeakyReLU(alpha=ALPHA)(x)
        
        self.nn = keras.Model(inputs=inputs, outputs=x, name="DDDQN Model")
        
        # use mean squared error loss and Adam optimizer
        self.nn.compile(loss=keras.losses.mean_squared_error,
                        optimizer='adam', metrics=['accuracy'])
        
    def copy_weights(self, other):
        # copies weights from this neural network to the another network
        for main_layer, other_layer in zip(self.nn.layers, other.nn.layers):
            weights = main_layer.get_weights()
            other_layer.set_weights(weights)
            
    def predict_one(self, board):
        return self.nn.predict(np.array(board, ndmin=2))
    
    def predict_batch(self, boards):
        return self.nn.predict(boards)
    
    def train_batch(self, x_batch, y_batch, batch, use_fit=True):
        if use_fit:
            return self.nn.fit(x_batch, y_batch, verbose=0, batch_size=batch,epochs=1)
        else:
            return self.nn.train_on_batch(x_batch, y_batch)
    
if __name__ == '__main__':
    model = Model(9, 9, 100, False)
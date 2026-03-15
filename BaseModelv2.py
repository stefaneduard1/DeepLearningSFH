from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

class StarNet2017:
    #Note in the paper they used filter length 20 and max pooling length of 10
    def __init__(self):
        self._model_type = 'StarNet2017CNN'
        self.lr = 0.0007
        self.initializer = 'he_normal'
        self.activation = 'relu'
        self.num_filters = [4, 16]
        self.filter_len = 20
        self.pool_length = 10
        self.num_hidden = [256, 128]
        self.l2 = 0
        self.optimizer = Adam(learning_rate=self.lr)
        self.last_layer_activation = 'linear'

    def model(self, N_wavelength_pixels):
        input_tensor = Input(shape=(N_wavelength_pixels, 2)) #2 inputs flux and noise spectrum
        cnn_layer_1 = Conv1D(kernel_initializer=self.initializer, activation=self.activation, padding="same",
                             filters=self.num_filters[0], kernel_size=self.filter_len,
                             kernel_regularizer=l2(self.l2), bias_regularizer=l2(self.l2))(input_tensor)
        cnn_layer_2 = Conv1D(kernel_initializer=self.initializer, activation=self.activation, padding="same",
                             filters=self.num_filters[1], kernel_size=self.filter_len,
                             kernel_regularizer=l2(self.l2), bias_regularizer=l2(self.l2))(cnn_layer_1)
        maxpool_1 = MaxPooling1D(pool_size=self.pool_length)(cnn_layer_2)
        flattener = Flatten()(maxpool_1)
        #3 Dense layers (Fully connected layers)
        layer_3 = Dense(units=self.num_hidden[0], kernel_initializer=self.initializer, activation=self.activation,
                        kernel_regularizer=l2(self.l2), bias_regularizer=l2(self.l2))(flattener)
        layer_4 = Dense(units=self.num_hidden[1], kernel_initializer=self.initializer, activation=self.activation,
                        kernel_regularizer=l2(self.l2), bias_regularizer=l2(self.l2))(layer_3)
        layer_out = Dense(units=4, kernel_initializer=self.initializer,
                          activation=self.last_layer_activation, name='output')(
            layer_4)
        model = Model(inputs=input_tensor, outputs=layer_out)
        model.compile(optimizer=self.optimizer, loss='mse')
        
        return model

    

  # adding deeper layer (num_filters) gets 32, 64 added to array deeper filters to hopefully find more complex spectral features (not sure if this will help our age guestimation though maybe it helps metallicity)
  # Lowered Pool length to 5 to not lose wavelength resolution, reducing rate at which spectral resolution is lost
  # Added one extra Conv layer in block one with num_filters 32
  # Add another block that has conv layers with num_filters 32 and 64, to try learn more global structures
  # For num_filters 32 or above kernel_size lowered to 5
  # GlobalAveragePooling instead of flatten to prevent overfitting now that we have added more layers
  # Added Dropout, to prevent it from overusing certain activations
  # Changed the learning rate to be 0.0003
  
  # Could still add: residual connection? 
  # Batchnormalization could cause slower convergence per epoch
  # Dropout might be too high (over regularization thus underfitting)

  # If it now overfits set dropout lower or we could try the other loss function again but most likely too much regularizers with Batchnorm and dropout included
  # If underfitting we might want to try just batchnorm or dropout?

  # If no improvement, the data preprocessing might be the problem? Though not sure how since they already did that pretty extensively 
  # Also if we have time skip connection???
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input, Concatenate, GlobalAveragePooling1D, BatchNormalization, Activation, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

class StarNet2017_DeeperNetwork:
    #Note in the paper they used filter length 20 and max pooling length of 10
    def __init__(self):
        self._model_type = 'StarNet2017CNN'
        self.lr = 0.0003
        self.initializer = 'he_normal'
        self.activation = 'relu'
        self.num_filters = [4, 8, 16, 32]
        self.filter_len = [20, 5]
        self.pool_length = 5
        self.num_hidden = [256, 128]
        self.l2 = 0
        self.optimizer = Adam(learning_rate=self.lr)
        self.last_layer_activation = 'linear'

    def model(self, N_wavelength_pixels, units=4):
        input_tensor = Input(shape=(N_wavelength_pixels, 2)) #2 inputs flux and noise spectrum

        #Block 1
        cnn_layer_1 = Conv1D(kernel_initializer=self.initializer, activation=None, padding="same",
                             filters=self.num_filters[0], kernel_size=self.filter_len[0],
                             kernel_regularizer=l2(self.l2), bias_regularizer=l2(self.l2))(input_tensor)
        cnn_layer_1_batch = BatchNormalization()(cnn_layer_1)
        cnn_layer_1_act = Activation(self.activation)(cnn_layer_1_batch)

        cnn_layer_2 = Conv1D(kernel_initializer=self.initializer, activation=None, padding="same",
                             filters=self.num_filters[1], kernel_size=self.filter_len[0],
                             kernel_regularizer=l2(self.l2), bias_regularizer=l2(self.l2))(cnn_layer_1_act)
        cnn_layer_2_batch = BatchNormalization()(cnn_layer_2)
        cnn_layer_2_act = Activation(self.activation)(cnn_layer_2_batch)

        cnn_layer_3 = Conv1D(kernel_initializer=self.initializer, activation=None, padding="same",
                             filters=self.num_filters[2], kernel_size=self.filter_len[1],
                             kernel_regularizer=l2(self.l2), bias_regularizer=l2(self.l2))(cnn_layer_2_act)
        cnn_layer_3_batch = BatchNormalization()(cnn_layer_3)
        cnn_layer_3_act = Activation(self.activation)(cnn_layer_3_batch)

        maxpool_1 = MaxPooling1D(pool_size=self.pool_length)(cnn_layer_3_act)

        # Adding a skip after block 1
        skip = GlobalAveragePooling1D()(maxpool_1)  # We will feed this also into the dense layers.

        #Block 2
        cnn_layer_4 = Conv1D(kernel_initializer=self.initializer, activation=None, padding="same",
                             filters=self.num_filters[2], kernel_size=self.filter_len[1],
                             kernel_regularizer=l2(self.l2), bias_regularizer=l2(self.l2))(maxpool_1)
        cnn_layer_4_batch = BatchNormalization()(cnn_layer_4)
        cnn_layer_4_act = Activation(self.activation)(cnn_layer_4_batch)

        cnn_layer_5 = Conv1D(kernel_initializer=self.initializer, activation=None, padding="same",
                             filters=self.num_filters[3], kernel_size=self.filter_len[1],
                             kernel_regularizer=l2(self.l2), bias_regularizer=l2(self.l2))(cnn_layer_4_act)
        cnn_layer_5_batch = BatchNormalization()(cnn_layer_5)
        cnn_layer_5_act = Activation(self.activation)(cnn_layer_5_batch)

        maxpool_2 = MaxPooling1D(pool_size=self.pool_length)(cnn_layer_5_act)

        gap = GlobalAveragePooling1D()(maxpool_2)
        merged = Concatenate()([gap, skip])  # The sizes are a bit weird but it should be fine to just concatenate these.

        #3 Dense layers (Fully connected layers)
        layer_3 = Dense(units=self.num_hidden[0], kernel_initializer=self.initializer, activation=self.activation,
                        kernel_regularizer=l2(self.l2), bias_regularizer=l2(self.l2))(merged)
        layer_3_drop = Dropout(0.15)(layer_3)


        layer_4 = Dense(units=self.num_hidden[1], kernel_initializer=self.initializer, activation=self.activation,
                        kernel_regularizer=l2(self.l2), bias_regularizer=l2(self.l2))(layer_3_drop)
        layer_4_drop = Dropout(0.15)(layer_4)

        layer_out = Dense(units=units, kernel_initializer=self.initializer,
                          activation=self.last_layer_activation, name='output')(layer_4_drop)

        model = Model(inputs=input_tensor, outputs=layer_out)
        model.compile(optimizer=self.optimizer, loss='mse')

        return model



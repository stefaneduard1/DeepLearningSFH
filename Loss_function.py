from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import keras.ops as k

# Loss function
def custom(y_true, y_pred):

    loss = 0

    for i in range(4):

        y_t = y_true[:, i]
        y_p = y_pred[:, 2*i]

        sigma = k.softplus(y_pred[:, 2*i+1]) + 1e-3

        loss += ((y_t - y_p)/sigma)**2 + 2*k.log(sigma)

    return k.mean(loss)

#Not sure what this is? If  this is different 
class StarNet2026:
    #Note in the paper they used filter length 20 and max pooling length of 10
    #This version if for the custom loss fucntion
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
        layer_out = Dense(units=8, kernel_initializer=self.initializer,
                          activation=self.last_layer_activation, name='output')(
            layer_4)
        model = Model(inputs=input_tensor, outputs=layer_out)

        return model


import tarfile
import numpy as np
from astropy.io import fits
import os 

print(os.getcwd())



tar_path = "C:\\Users\\Stefan\\Desktop\\Deep Learning\\Project\\Data\\MockSpectra-Woo2024\\v1_training_spectra\\bin000.tar.gz"

spectra = []
noise_list = []

with tarfile.open(tar_path, "r") as tar:

    members = tar.getmembers()
    
    for index, member in enumerate(members):
        # if not index%100:
        print(index)
        file = tar.extractfile(member)
        with fits.open(file) as hdu:

            spec = hdu[1].data["spec"]
            var = hdu[1].data["var"]

            # noise = np.sqrt(var)

            # spectra.append(spec)
            # noise_list.append(noise)

spectra = np.array(spectra)
noise = np.array(noise_list)

X = np.stack([spectra, noise], axis=-1) #Convert it to the spectra + noise (1000, N_wavelengths, 2) shape

print("Spectra shape:", spectra.shape)
print("Noise shape:", noise.shape)

from astropy.table import Table
tab = Table.read("C:/Users/Stefan/Desktop/Deep Learning/Project/Data/MockSpectra-Woo2024/v1_training_spectra/datatab_Woo2024_training.fits.gz")



print(tab.colnames)

labels = []

for i in range(1000):

    fname = f"spec-{i}.fits"
    row = tab[tab["fname"] == fname][0]

    labels.append([
        row["logage_in"],
        row["metal_in"],
        row["ebv_in"],
        row["ML_r"]
    ])

y = np.array(labels)

print(y.shape)

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print(X_train.shape, y_train.shape)

model_builder = StarNet2026()
model = model_builder.model(X.shape[1])

model.compile(
    optimizer=model_builder.optimizer,
    loss=custom
)


history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=1000,
    batch_size=32
)

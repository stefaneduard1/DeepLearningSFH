from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import keras.ops as k

#Loss function
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


import numpy as np
from astropy.io import fits
import os 

unpacked_path = "C:/Users/Stefan/Desktop/Deep Learning/Project/Data/MockSpectra-Woo2024/v1_training_spectra_extracted"

# ── Control how many bin folders to read ──────────────────────────────────────
NUM_FOLDERS = 25  # change this to test with more or fewer folders

N_PIXELS = 4544
N_PER_FOLDER = 1000
total_files = NUM_FOLDERS * N_PER_FOLDER

print(f"Pre-allocating arrays for {total_files} spectra of {N_PIXELS} pixels...")

# Pre-allocate arrays

# spectra = np.empty((total_files, N_PIXELS), dtype=np.float32)
# noise_list = np.empty((total_files, N_PIXELS), dtype=np.float32)

all_spectra = []
all_noise = []

# ──────────────────────────────────────────────────────────────────────────────

bin_folders = sorted([
    f for f in os.listdir(unpacked_path)
    if os.path.isdir(os.path.join(unpacked_path, f))
])[:NUM_FOLDERS]
 
print(f"Reading {len(bin_folders)} folder(s): {bin_folders}\n")
 
# for folder in bin_folders:
#     folder_path = os.path.join(unpacked_path, folder)
#     fits_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".fits")])
 
#     print(f"  [{folder}] Found {len(fits_files)} FITS files...")
 
#     for filename in fits_files:
#         filepath = os.path.join(folder_path, filename)
#         with fits.open(filepath) as hdu:
#             spec = hdu[1].data["spec"]
#             var = hdu[1].data["var"]
#             spectra.append(spec)
#             noise_list.append(np.sqrt(var))

# idx = 0
# for folder in bin_folders:
#     folder_path = os.path.join(unpacked_path, folder)
#     fits_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".fits")])
#     print(f"  [{folder}] Found {len(fits_files)} FITS files...")
#     for filename in fits_files:
#         filepath = os.path.join(folder_path, filename)
#         with fits.open(filepath, memmap=False) as hdu:
#             spectra[idx] = hdu[1].data["spec"]
#             noise_list[idx] = np.sqrt(hdu[1].data["var"])
#             idx += 1

# spectra = np.array(spectra)
# noise_list = np.array(noise_list)

for folder in bin_folders:
    folder_path = os.path.join(unpacked_path, folder)
    fits_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".fits")])
    print(f"  [{folder}] Loading...")

    bin_spectra = np.empty((N_PER_FOLDER, N_PIXELS), dtype=np.float32)
    bin_noise = np.empty((N_PER_FOLDER, N_PIXELS), dtype=np.float32)

    for i, filename in enumerate(fits_files):
        filepath = os.path.join(folder_path, filename)
        with fits.open(filepath, memmap=False) as hdu:
            bin_spectra[i] = hdu[1].data["spec"]
            bin_noise[i] = np.sqrt(hdu[1].data["var"])

    print("loaded, appending...")
    all_spectra.append(bin_spectra)
    all_noise.append(bin_noise)

all_spectra = np.concatenate(all_spectra, axis=0)
all_noise_list = np.concatenate(all_noise, axis=0)

print(f"\nDone! Loaded {len(all_spectra)} spectra total.")

X = np.stack([all_spectra, all_noise_list], axis=-1) #Convert it to the spectra + noise (1000, N_wavelengths, 2) shape

print("Spectra shape:", all_spectra.shape)
print("Noise shape:", all_noise.shape)
 
print(f"\nDone! Loaded {len(all_spectra)} spectra total.")

# import pandas as pd

# # peek at the first spectrum
# df = pd.DataFrame({
#     "spec": spectra[0],
#     "noise": noise_list[0]
# })p

# print(df.head(20))
# print(f"\nShape: {df.shape}")
# print(f"\nBasic stats:\n{df.describe()}")





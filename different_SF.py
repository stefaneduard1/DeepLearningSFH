from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint
import keras.ops as k

# Loss function
def custom(y_true, y_pred):

    loss = 0

    for i in range(4): 

        y_t = y_true[:, i]
        y_p = y_pred[:, 2*i]

        sigma = k.softplus(y_pred[:, 2*i+1]) + 0.1

        loss += ((y_t - y_p)/sigma)**2 + 2*k.log(sigma)

    return k.mean(loss)

# Not sure what this is? If  this is different 
class StarNet2026:
    # Note in the paper they used filter length 20 and max pooling length of 10
    # This version is for the custom loss fucntion
    def __init__(self):
        self._model_type = 'StarNet2017CNN'
        self.lr = 0.00005
        self.initializer = 'he_normal'
        self.activation = 'relu'
        self.num_filters = [4, 16]
        self.filter_len = 20
        self.pool_length = 10
        self.num_hidden = [256, 128]
        self.l2 = 0
        self.optimizer = Adam(learning_rate=self.lr)
        self.last_layer_activation = 'linear'

    def model(self, N_wavelength_pixels, units=4):
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
        layer_out = Dense(units=units, kernel_initializer=self.initializer,
                          activation=self.last_layer_activation, name='output')(
            layer_4)
        model = Model(inputs=input_tensor, outputs=layer_out)

        return model


import numpy as np
from astropy.io import fits
import os 

# unpacked_path = "/mnt/c/Users/Stefan/Desktop/Deep Learning/Project/Data/MockSpectra-Woo2024/v1_training_spectra_extracted"
unpacked_path = "/root/data/MockSpectra-Woo2024/v1_training_spectra_extracted"


# ── Control how many bin folders to read ──────────────────────────────────────
NUM_FOLDERS = 90  # change this to test with more or fewer folders



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
    print(len(all_spectra), len(all_noise))

all_spectra = np.array(np.concatenate(all_spectra, axis=0))
all_noise_list = np.array(np.concatenate(all_noise, axis=0))

print(f"\nDone! Loaded {len(all_spectra)} spectra total.")

X = np.stack([all_spectra, all_noise_list], axis=-1) #Convert it to the spectra + noise (1000, N_wavelengths, 2) shape

print("Spectra shape:", all_spectra.shape)
print("Noise shape:", all_noise_list.shape)
 
print(f"\nDone! Loaded {len(all_spectra)} spectra total.")

# Create y

from astropy.table import Table
# tablepath = "/mnt/c/Users/Stefan/Desktop/Deep Learning/Project/Data/MockSpectra-Woo2024/v1_training_spectra_extracted/datatab.fits"
tablepath = "/root/data/MockSpectra-Woo2024/v1_training_spectra_extracted/datatab_Woo2024_training.fits"


tab = Table.read(tablepath) #Read content of table for labels
#Give labels to values (this is just an example for the one bin (have to generalize if we want to do more)
fname_to_idx = {row['fname']: i for i, row in enumerate(tab)}

labels = []
num_spectra = NUM_FOLDERS * 1000

for i in range(num_spectra):
    fname = f"spec-{i}.fits"
    row = tab[fname_to_idx[fname]]
    labels.append([
        row["logage_in"],
        row["metal_in"],
        row["ebv_in"],
        row["ML_r"]
    ])

y = np.array(labels)

#Split into test and validation data 
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

#Compile the model (make sure to do this in a seperate cell in colab or wherever we run this
model_builder = StarNet2026()
model = model_builder.model(X.shape[1], units=8) #Uses the length of the wavelength 

model.compile(
    optimizer=model_builder.optimizer,
    loss=custom
)

callbacks = [
    ModelCheckpoint(
        "/mnt/c/Users/Stefan/Desktop/starnet_best_custom.keras",
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=35, #Change epochs to whatever we want 
    batch_size=32,
    verbose=1
)

import json
with open("/mnt/c/Users/Stefan/Desktop/history_custom.json", "w") as f:
    json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f)

model.save("/mnt/c/Users/Stefan/Desktop/starnet_model_custom.keras")


import matplotlib.pyplot as plt

# Get predictions on validation set
y_pred_raw = model.predict(x_val)

# The model outputs 8 values: [mu_0, sigma_0, mu_1, sigma_1, ...]
# So predicted means are at even indices
logage_pred = y_pred_raw[:, 0]   # mu for logage_in
metal_pred  = y_pred_raw[:, 1]   # mu for metal_in

logage_true = y_val[:, 0]
metal_true  = y_val[:, 1]

# fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# for ax, true, pred, label in zip(
#     axes,
#     [logage_true, metal_true],
#     [logage_pred, metal_pred],
#     ['logage_in', 'metal_in']
# ):
#     ax.scatter(true, pred, alpha=0.3, s=5, color='steelblue')
    
#     # 1:1 line
#     lims = [min(true.min(), pred.min()), max(true.max(), pred.max())]
#     ax.plot(lims, lims, 'r--', linewidth=1.5, label='1:1')
    
#     # Residual stats
#     residuals = pred - true
#     ax.set_title(f'{label}\nbias={residuals.mean():.3f}, std={residuals.std():.3f}')
#     ax.set_xlabel('True')
#     ax.set_ylabel('Predicted')
#     ax.legend()

# plt.suptitle('Predicted vs True (validation set)', fontsize=13)
# plt.tight_layout()
# plt.savefig("/mnt/c/Users/Stefan/Desktop/plot.png")


fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(history.history['loss'], label='train')
axes[0].plot(history.history['val_loss'], label='val')
axes[0].set_title('Loss over epochs')
axes[0].set_xlabel('Epoch')
axes[0].legend()

axes[1].plot(history.history['loss'], label='train')
axes[1].plot(history.history['val_loss'], label='val')
axes[1].set_title('Loss over epochs (log scale)')
axes[1].set_xlabel('Epoch')
axes[1].set_ylim(-15, 30)  # clips out the 283k and 45k spikes
axes[1].legend()

plt.tight_layout()
plt.savefig("/mnt/c/Users/Stefan/Desktop/training_curves_custom.png")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for idx, (ax, col, pred_col, label) in enumerate(zip(
    axes.flat,
    [0, 1, 2, 3],      # true cols
    [0, 2, 4, 6],      # pred mean cols
    ['logage_in', 'metal_in', 'ebv_in', 'ML_r']
)):
    true = y_val[:, col]
    pred = y_pred_raw[:, pred_col]
    residuals = pred - true

    ax.scatter(true, pred, alpha=0.3, s=5, color='steelblue')
    lims = [min(true.min(), pred.min()), max(true.max(), pred.max())]
    ax.plot(lims, lims, 'r--', linewidth=1.5)
    ax.set_title(f'{label}\nbias={residuals.mean():.3f}, std={residuals.std():.3f}')
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')

plt.suptitle('Predicted vs True (validation set)', fontsize=13)
plt.tight_layout()
plt.savefig("/mnt/c/Users/Stefan/Desktop/pred_vs_true_custom.png")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for idx, (ax, col, label) in enumerate(zip(
    axes.flat,
    [0, 1, 2, 3],
    ['logage_in', 'metal_in', 'ebv_in', 'ML_r']
)):
    residuals = y_pred_raw[:, col] - y_val[:, col]
    ax.hist(residuals, bins=100, color='steelblue', edgecolor='none')
    ax.axvline(0, color='r', linestyle='--')
    ax.set_title(f'{label}\nbias={residuals.mean():.3f}, std={residuals.std():.3f}')
    ax.set_xlabel('Residual (pred - true)')

plt.suptitle('Residual distributions (validation set)', fontsize=13)
plt.tight_layout()
plt.savefig("/mnt/c/Users/Stefan/Desktop/residuals_custom.png")



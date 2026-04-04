import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from astropy.io import fits
from astropy.table import Table
import os
import keras.ops as k

# ── Custom loss ───────────────────────────────────────────────────────────────
def custom(y_true, y_pred):

    loss = 0
    # Parameters are log age, metallicity, color, and mass/light ratio.
    # We want to weight metallicity higher. 
    weights = [1., 3., 1., 1.]

    for i in range(4):

        y_t = y_true[:, i]
        y_p = y_pred[:, 2*i]

        sigma = k.softplus(y_pred[:, 2*i+1]) + 0.1
        
        target_loss = ((y_t - y_p)/sigma)**2 + 2*k.log(sigma) + 0.1 * sigma
        loss += weights[i] * target_loss

    return k.mean(loss)

# ── Config ────────────────────────────────────────────────────────────────────
model_path    = "/mnt/c/Users/Stefan/Desktop/starnet_model_custom.keras"
unpacked_path = "/root/data/MockSpectra-Woo2024/v1_training_spectra_extracted"
tablepath     = "/root/data/MockSpectra-Woo2024/v1_training_spectra_extracted/datatab_Woo2024_training.fits"
NUM_FOLDERS   = 90
N_PIXELS      = 4544
N_PER_FOLDER  = 1000

# ── Load model ────────────────────────────────────────────────────────────────
model = load_model(model_path, custom_objects={"custom": custom})

# ── Load table ────────────────────────────────────────────────────────────────
tab = Table.read(tablepath)
fname_to_idx = {row['fname']: i for i, row in enumerate(tab)}

# ── Get bin folders ───────────────────────────────────────────────────────────
bin_folders = sorted([
    f for f in os.listdir(unpacked_path)
    if os.path.isdir(os.path.join(unpacked_path, f))
])[:NUM_FOLDERS]

# ── Accumulate predictions bin by bin ────────────────────────────────────────
all_preds = []
all_true  = []

from astropy.table import Table
# tablepath = "/mnt/c/Users/Stefan/Desktop/Deep Learning/Project/Data/MockSpectra-Woo2024/v1_training_spectra_extracted/datatab.fits"
tablepath = "/root/data/MockSpectra-Woo2024/v1_training_spectra_extracted/datatab_Woo2024_training.fits"

fyoung_min, fyoung_max = [0., 1]  # first bin
# fyoung_min, fyoung_max = [1e-7, 1e-2]  # second bin
# fyoung_min, fyoung_max = [1e-2, 1]  # third bin

fulltable = Table.read(tablepath) #Read content of table for labels
tab = fulltable[:1000*NUM_FOLDERS]

fyoungs = tab['fyoung']

mask = ((fyoungs >= fyoung_min) & (fyoungs <= fyoung_max)).astype(int)

for index, folder in enumerate(bin_folders):
    print(f"Predicting [{folder}]...")
    folder_path = os.path.join(unpacked_path, folder)
    fits_files  = sorted([f for f in os.listdir(folder_path) if f.endswith(".fits")])

    slicedmask = mask[index*1000:(index+1)*1000]

    bin_spectra = np.empty((np.sum(slicedmask), N_PIXELS), dtype=np.float32)
    bin_noise   = np.empty((np.sum(slicedmask), N_PIXELS), dtype=np.float32)
    bin_labels  = []

    j = 0
    for i, filename in enumerate(fits_files):
        if slicedmask[i]:
            filepath = os.path.join(folder_path, filename)
            with fits.open(filepath, memmap=False) as hdu:
                bin_spectra[i-j] = hdu[1].data["spec"]
                bin_noise[i-j]   = np.sqrt(hdu[1].data["var"])
            row = tab[fname_to_idx[filename]]
            bin_labels.append([row["logage_in"], row["metal_in"], row["ebv_in"], row["ML_r"]])
        else:
            j += 1

    X_bin = np.stack([bin_spectra, bin_noise], axis=-1)
    preds = model.predict(X_bin, verbose=0)

    all_preds.append(preds)
    all_true.append(np.array(bin_labels))

    del X_bin, bin_spectra, bin_noise  # free immediately

all_preds = np.concatenate(all_preds, axis=0)
all_true  = np.concatenate(all_true,  axis=0)

print("Done predicting! Plotting...")

# ── Plot ──────────────────────────────────────────────────────────────────────
labels     = ['logage_in', 'metal_in', 'ebv_in', 'ML_r']
true_cols  = [0, 1, 2, 3]
pred_cols  = [0, 2, 4, 6]
# sigma_cols = [1, 3, 5, 7]

# Pred vs True
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
limits = [[8.65, 10.55], [-0.65, 0.25], [-0.05, 2.6], [0., 4.2]]
for ax, tc, pc, label, lim in zip(axes.flat, true_cols, pred_cols, labels, limits):
    true = all_true[:, tc]
    pred = all_preds[:, pc]
    residuals = pred - true
    ax.scatter(true, pred, alpha=0.3, s=5, color='steelblue')
    lims = lim
    ax.plot(lims, lims, 'r--', linewidth=1.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title(f'{label}\nbias={residuals.mean():.3f}, std={residuals.std():.3f}')
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
plt.suptitle('Predicted vs True', fontsize=13)
plt.tight_layout()
plt.savefig("/mnt/c/Users/Stefan/Desktop/pred_vs_true_custom.png")

limits = [[-2, 2], [-1, 1], [-0.5, 0.5], [-2.5, 2.5]]

# Residuals
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, tc, pc, lim, label in zip(axes.flat, true_cols, pred_cols, limits, labels):
    bins = np.linspace(lim[0], lim[1], 100)
    residuals = all_preds[:, pc] - all_true[:, tc]
    ax.hist(residuals, bins=bins, color='steelblue', edgecolor='none')
    ax.axvline(0, color='r', linestyle='--')
    ax.set_title(f'{label}\nbias={residuals.mean():.3f}, std={residuals.std():.3f}')
    ax.set_xlabel('Residual (pred - true)')
    ax.set_xlim(lim)
    # ax.set_xscale('log')
    # ax.set_xlim(1e-3, 1e+3)
    # ax.set_yscale('log')
plt.suptitle('Residual distributions', fontsize=13)
plt.tight_layout()
plt.savefig("/mnt/c/Users/Stefan/Desktop/residuals_custom.png")

limits = [[-10, 10], [-10, 10], [-20, 20], [-5, 5]]

# # Sigmas
# fig, axes = plt.subplots(2, 2, figsize=(12, 10))
# for ax, sc, lim, label in zip(axes.flat, sigma_cols, limits, labels):
#     bins = np.linspace(lim[0], lim[1], 100)
#     ax.hist(all_preds[:, sc], bins=bins, color='coral', edgecolor='none')
#     ax.set_title(f'{label} — predicted σ')
#     ax.set_xlabel('σ')
#     ax.set_xlim(lim)
#     # ax.set_xscale('log')
#     # ax.set_xlim(1e-3, 1e+3)
#     # ax.set_yscale('log')
# plt.suptitle('Predicted uncertainties', fontsize=13)
# plt.tight_layout()
# plt.savefig("/mnt/c/Users/Stefan/Desktop/sigmas_custom.png")

# Residuals vs True
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
limits_true = [[8.65, 10.55], [-0.65, 0.25], [-0.05, 2.6], [0., 4.2]]
limits_res  = [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]]
for ax, tc, pc, lim_t, lim_r, label in zip(axes.flat, true_cols, pred_cols, limits_true, limits_res, labels):
    true      = all_true[:, tc]
    residuals = all_preds[:, pc] - true
    ax.scatter(true, residuals, alpha=0.3, s=5, color='steelblue')
    ax.axhline(0, color='r', linestyle='--', linewidth=1.5)
    ax.set_xlim(lim_t)
    ax.set_ylim(lim_r)
    ax.set_title(f'{label}\nbias={residuals.mean():.3f}, std={residuals.std():.3f}')
    ax.set_xlabel('True')
    ax.set_ylabel('Residual (pred - true)')
plt.suptitle('Residuals vs True', fontsize=13)
plt.tight_layout()
plt.savefig("/mnt/c/Users/Stefan/Desktop/residuals_vs_true_custom.png")

print("All plots saved to Desktop!")

hist = {
    'loss': [9816.7061, 24.7427, 18.1834, 91.4551, 25.3859, 15.8660, 66.1695, 118.6482, 15.0645, -1.1881, -7.4518, -5.3526, 565.5087, 9.7271, -7.0931, -9.7590, -10.0874, -10.3263, -1.3184, -9.2830, 942.6281, 0.3461, -3.9306, -5.7716, -2.9853, 359.8432, 1.1268, -6.0560, -10.3305, -8.5246, 3683.1125, -5.1563, -10.5702, -9.0028, -9.9058],
    'val_loss': [24.5922, 22.1395, 47.3883, 23.2454, 22.3274, 7.7964, 283217.9688, 22.3550, 3.2382, -2.2388, -8.0449, -8.1223, 16.1218, -1.2973, -8.2911, -9.0381, -9.0154, -9.4841, -3.7232, 45508.7031, 14.8793, -3.2586, -4.9722, -1.8538, -4.0685, 2.3218, -1.2546, -9.9888, -9.7401, -10.1897, -2.5450, -9.6782, -11.0283, -3.0219, 129.3897]
}

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Full picture
axes[0].plot(hist['loss'], label='train')
axes[0].plot(hist['val_loss'], label='val')
axes[0].set_title('Loss over epochs (full)')
axes[0].set_xlabel('Epoch')
axes[0].legend()

# Clipped view — ignore the explosion spikes
axes[1].plot(hist['loss'], label='train')
axes[1].plot(hist['val_loss'], label='val')
axes[1].set_title('Loss (clipped view)')
axes[1].set_xlabel('Epoch')
axes[1].set_ylim(-15, 30)  # clips out the 283k and 45k spikes
axes[1].legend()

plt.tight_layout()
plt.savefig("/mnt/c/Users/Stefan/Desktop/training_curves.png")
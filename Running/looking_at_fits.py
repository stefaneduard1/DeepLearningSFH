import numpy as np 
from astropy.io import fits
from astropy.table import Table

path = "/root/data/MockSpectra-Woo2024/v1_training_spectra_extracted/datatab_Woo2024_training.fits"

# First, inspect the file structure
with fits.open(path) as hdul:
    hdul.info()  # Shows all HDUs (extensions)
    print("\n--- Header of extension 1 ---")
    print(hdul[1].header)

# Try reading as a Table
tab = Table.read(path)
tab = tab[:90000]

print("\n--- Column names ---")
print(tab.colnames)

print("HERE!!!!")
print(tab['fname'])

print("\n--- Shape / length ---")
print(f"Rows: {len(tab)}")

print("\n--- First 5 rows ---")
print(tab[:5])

import matplotlib.pyplot as plt

fyoung = tab['fyoung']
print("HELLO HELLO",fyoung[fyoung > 0].min())

# plt.figure(figsize=(8, 5))
# plt.hist(fyoung, bins=1000, color='steelblue', edgecolor='none', alpha=0.8)
# plt.xlabel('fyoung')
# plt.ylabel('Count')
# plt.xlim(-0.1, 0.3)
# plt.title('Distribution of fyoung (n=250,000)')
# plt.tight_layout()
# plt.show()

hor = tab['fyoung']

# fig = plt.figure()
# frame = fig.add_subplot(1,1,1)
# frame.scatter(hor, avgssfr)
# frame.set_xlabel("fyoung")
# frame.set_ylabel('avgssfr')
# frame.set_xlim(-0.05,0.6)
# frame.set_ylim(-0.05,0.6)
# plt.show()

order = np.argsort(tab['fyoung'])
sorted_fyoung = tab['fyoung'][order]

i=0
while sorted_fyoung[i] == 0:
    i+= 1

threshold1 = i
while sorted_fyoung[i] < 1e-2:
    i+= 1

threshold2 = i

print(threshold1, threshold2)

names = tab['fname']
names_quiescent = names[order[:threshold1]]
names_greenvalley = names[order[threshold1:threshold2]]
names_starforming = names[order[threshold2:]]

print(names_quiescent, names_greenvalley, names_starforming)

# print(len(sorted_fyoung))

# plt.figure(figsize=(8, 5))
# plt.hist(sorted_fyoung[:cut1], bins=1000, color='steelblue', edgecolor='none', alpha=0.8)
# plt.xlabel('fyoung')
# plt.ylabel('Count')
# plt.title('Distribution of fyoung (n=250,000)')
# plt.tight_layout()
# plt.show()

# print(np.sum(sorted_fyoung[:cut1]))
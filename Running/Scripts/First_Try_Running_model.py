import tarfile
import numpy as np
from astropy.io import fits

#Each bin contains 1000 spectra
tar_path = "/bin000.tar.gz" #Whatever is the correct path, can generalise for more bins 

spectra = []
noise_list = []

with tarfile.open(tar_path, "r:gz") as tar:

    members = tar.getmembers()

    for member in members:

        file = tar.extractfile(member)
        hdu = fits.open(file)

        spec = hdu[1].data["spec"]
        var = hdu[1].data["var"]

        noise = np.sqrt(var)

        spectra.append(spec)
        noise_list.append(noise)

spectra = np.array(spectra)
noise = np.array(noise_list)

x = np.stack([spectra, noise], axis=-1) #Convert it to the spectra + noise (1000, N_wavelengths, 2) shape

from astropy.table import Table

tab = Table.read("/datatab_Woo2024_training.fits.gz") #Read content of table for labels
#Give labels to values (this is just an example for the one bin (have to generalize if we want to do more)
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



#Split into test and validation data 
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

#Compile the model (make sure to do this in a seperate cell in colab or wherever we run this
model_builder = StarNet2017()
model = model_builder.model(x.shape[1]) #Uses the length of the wavelength 

model.compile(
    optimizer=model_builder.optimizer,
    loss="mse"
)
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=10, #Change epochs to whatever we want 
    batch_size=32
)



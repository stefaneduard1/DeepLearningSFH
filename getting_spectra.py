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

X = np.stack([spectra, noise], axis=-1) #Convert it to the spectra + noise (1000, N_wavelengths, 2) shape

print("Spectra shape:", spectra.shape)
print("Noise shape:", noise.shape)

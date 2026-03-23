import tarfile
import os
 
base_path = "C:\\Users\\Stefan\\Desktop\\Deep Learning\\Project\\Data\\MockSpectra-Woo2024\\v1_training_spectra\\"
extract_base = "C:\\Users\\Stefan\\Desktop\\Deep Learning\\Project\\Data\\MockSpectra-Woo2024\\v1_training_spectra_extracted\\"
 
os.makedirs(extract_base, exist_ok=True)
 
tar_files = [f for f in os.listdir(base_path) if f.endswith(".tar.gz")]
total = len(tar_files)
 
print(f"Found {total} tar.gz files to extract.\n")
 
for i, filename in enumerate(sorted(tar_files), start=1):
    bin_name = filename.replace(".tar.gz", "")
    extract_path = os.path.join(extract_base, bin_name)
    os.makedirs(extract_path, exist_ok=True)
 
    print(f"[{i}/{total}] Extracting {filename}...")
    with tarfile.open(os.path.join(base_path, filename), "r") as tar:
        tar.extractall(extract_path)
 
print("\nAll done!")
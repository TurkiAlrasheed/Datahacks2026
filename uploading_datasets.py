import edgeimpulse as ei
import os
ei.API_KEY = "ei_13c0a8fc41e3ad5c23a7d591bffcdcdf97d6c163bd4f2817"

path = "path/to/dataset"
species = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

for s in species:
    ei.data.upload_directory(
        directory=f"{path}/{s}",
        category="training",
        label=s,
    )
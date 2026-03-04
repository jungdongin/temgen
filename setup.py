from setuptools import setup, find_packages

setup(
    name="temgen",
    version="0.1.0",
    description="TEM diffraction pattern + crystal structure contrastive learning",
    author="Dongin Kim",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "pymatgen>=2024.1.1",
        "h5py>=3.9",
        "zarr>=2.16,<3.0",
        "pytorch-lightning>=2.1",
        "torchmetrics>=1.2",
        "wandb>=0.16",
        "omegaconf>=2.3",
        "tqdm>=4.65",
        "einops>=0.7",
    ],
)
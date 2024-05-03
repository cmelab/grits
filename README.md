# GRiTS : Grits Reduces/Restores Topology with SMILES
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/grits/badges/version.svg)](https://anaconda.org/conda-forge/grits)
[![pytest](https://github.com/cmelab/grits/actions/workflows/pytest.yml/badge.svg)](https://github.com/cmelab/grits/actions/workflows/pytest.yml)
[![codecov](https://codecov.io/gh/cmelab/grits/branch/main/graph/badge.svg?token=lGG8Zf65HP)](https://codecov.io/gh/cmelab/grits)
[![Documentation Status](https://readthedocs.org/projects/grits/badge/?version=latest)](https://grits.readthedocs.io/en/latest/?badge=latest)
[![Docker build](https://github.com/cmelab/grits/actions/workflows/build.yml/badge.svg)](https://github.com/cmelab/grits/actions/workflows/build.yml)

![GRiTS workflow](/.github/grits.png)

GRiTS is a toolkit for working with coarse-grain systems. It uses [mBuild](https://github.com/mosdef-hub/mbuild) to build up molecules and [SMILES](https://www.daylight.com/dayhtml/doc/theory/theory.smiles.html) chemical grammar to facilitate the mapping from fine-to-coarse and coarse-to-fine.

### Installation
#### Using a container

To use GRiTS in a prebuilt container (using [Singularity](https://singularity.lbl.gov/)), run:
```bash
singularity pull docker://ghcr.io/cmelab/grits:0.4.1
singularity exec grits_0.4.1.sif bash
```

**Or** using [Docker](https://docs.docker.com/), run:
```bash
docker pull ghcr.io/cmelab/grits:0.4.1
docker run -it cmelab/grits:0.4.1
```

#### Custom install
To create a local environment with [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html)), run:
```bash
micromamba create -f environment.yml
micromamba activate grits
```
With the `grits` environment active, install the package with pip:
```
pip install .
```
And to test your installation, run:
```
pytest
```

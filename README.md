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

To use GRiTS in a prebuilt container (using [Apptainer](https://apptainer.org/)), run:
```bash
apptainer pull docker://ghcr.io/cmelab/grits:latest
apptainer run grits_latest.sif python
```

**Or** using [Docker](https://docs.docker.com/), run:
```bash
docker pull ghcr.io/cmelab/grits:latest
docker run -it cmelab/grits:latest
```

#### Micromamba install
To create a local environment with [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html), run:
```bash
micromamba create grits -f environment.yml
micromamba activate grits
```

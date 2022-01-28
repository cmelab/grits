# GRiTS : Grits Reduces/Restores Topology with SMILES
[![Docker Repository on Quay](https://quay.io/repository/cmelab/grits/status "Docker Repository on Quay")](https://quay.io/repository/cmelab/grits)
[![pytest](https://github.com/cmelab/grits/actions/workflows/pytest.yml/badge.svg)](https://github.com/cmelab/grits/actions/workflows/pytest.yml)
[![codecov](https://codecov.io/gh/cmelab/grits/branch/main/graph/badge.svg?token=lGG8Zf65HP)](https://codecov.io/gh/cmelab/grits)

![GRiTS workflow](/.github/grits.png)

GRiTS is a toolkit for working with coarse-grain systems. It uses [mBuild](https://github.com/mosdef-hub/mbuild) to build up molecules and [SMILES](https://www.daylight.com/dayhtml/doc/theory/theory.smiles.html) chemical grammar to facilitate the mapping from fine-to-coarse and coarse-to-fine.

### Installation
#### Using a container

To use GRiTS in a prebuilt container (using [Singularity](https://singularity.lbl.gov/)), run:
```bash
singularity pull docker://quay.io/cmelab/grits:latest
singularity exec grits_latest.sif bash
```

**Or** using [Docker](https://docs.docker.com/), run:
```bash
docker pull quay.io/cmelab/grits:latest
docker run -it cmelab/grits:latest
```

#### Custom install
To create a local environment with [conda](https://docs.conda.io/en/latest/miniconda.html), run:
```bash
conda env create -f environment.yml
conda activate grits
```
And to test your installation, run:
```
pytest
```

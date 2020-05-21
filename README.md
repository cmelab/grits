# GRiTS : Grit Reduces/Restores (i) Topology with SMILES
A toolkit for working with coarse-grain systems using mbuild and fresnel

Minimal install
```
conda create -n grit
conda activate grit
conda install -c conda-forge -c omnia -c mosdef pillow numpy matplotlib jupyterlab mbuild fresnel pyside2 freud openbabel gsd foyer hoomd
```

My install
```
conda create -n grit
conda activate grit
conda install -c conda-forge -c omnia -c mosdef pillow pytest sphinx sphinx_rtd_theme nbsphinx numpy matplotlib black isort jupyterlab mbuild fresnel pyside2 freud py3dmol openbabel gsd foyer hoomd
jupyter labextension install @ryantam626/jupyterlab_code_formatter
pip install jupyterlab_code_formatter
jupyter serverextension enable --py jupyterlab_code_formatter
```

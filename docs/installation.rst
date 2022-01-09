Installation
============

Installation of GRiTS uses the conda package manager. We recommend `Miniconda`_ or `Mamba`_.
All comands are run from the top deirectory of the GRiTS repository. The ``grits`` environment can be created using::

    conda env create -f environment.yml
    conda activate grits

With the ``grits`` environment active, the package can be installed using pip::

    pip install .

.. _Miniconda: https://docs.conda.io/en/latest/miniconda.html
.. _Mamba: https://github.com/mamba-org/mamba

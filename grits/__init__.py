"""GRiTS: A coarse-grain toolkit using chemical grammars."""

from importlib.metadata import PackageNotFoundError, version

from . import utils
from .coarsegrain import Bead, CG_Compound, CG_System
from .finegrain import backmap

try:
    __version__ = version("grits")
except PackageNotFoundError:
    # package is not installed
    pass

__all__ = [
    "Bead",
    "CG_Compound",
    "CG_System",
    "__version__",
    "backmap",
    "utils",
]

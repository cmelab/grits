#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Example copied with love from:
# https://github.com/kennethreitz/setup.py/blob/master/setup.py

# Note: To use the 'upload' functionality of this file, you must:
#   $ pip install twine

import os
import sys
from shutil import rmtree

from setuptools import Command, find_packages, setup

# Package meta-data.
NAME = "grits"
DESCRIPTION = "A toolkit for working with coarse-grained systems"
URL = "https://github.com/cmelab/grits"
EMAIL = "chrisjones4@u.boisestate.edu"
AUTHOR = "CME Lab, Jenny Fothergill"
REQUIRES_PYTHON = ">=3.9.0"
VERSION = None

# What packages are required for this module to be executed?
REQUIRED = []

# Follow the README and use the environment.yml file to install
# the needed packages.
# ------------------------------------------------

here = os.path.abspath(os.path.dirname(__file__))


def myversion():
    from setuptools_scm.version import get_local_dirty_tag

    def clean_scheme(version):
        return get_local_dirty_tag(version) if version.dirty else "+clean"

    return {"local_scheme": clean_scheme}


class UploadCommand(Command):
    """Support setup.py upload."""

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds…")
            rmtree(os.path.join(here, "dist"))
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution…")
        os.system(
            "{0} setup.py sdist bdist_wheel --universal".format(sys.executable)
        )

        self.status("Uploading the package to PyPi via Twine…")
        os.system("twine upload dist/*")

        self.status("Pushing git tags…")
        os.system("git tag v{0}".format(about["__version__"]))
        os.system("git push --tags")

        sys.exit()


# Where the magic happens:
setup(
    name=NAME,
    use_scm_version=myversion,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(
        exclude=(
            "tests",
            "docs",
        )
    ),
    package_data={"grits": ["compounds/*"]},
    install_requires=REQUIRED,
    include_package_data=True,
    license="GPLv3",
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    # $ setup.py publish support.
    cmdclass={
        "upload": UploadCommand,
    },
)

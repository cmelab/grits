from codecs import open
from os import path

from setuptools import find_packages, setup

# Package meta-data.
NAME = "grits"
DESCRIPTION = "A toolkit for working with coarse-grained systems"
URL = "https://github.com/cmelab/grits"
EMAIL = "jennyfothergill@boisestate.edu"
AUTHOR = "Jenny Fothergill"
REQUIRES_PYTHON = ">=3.8"

# What packages are required for this module to be executed?
REQUIRED = []

here = path.abspath(path.dirname(__file__))

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


def myversion():
    from setuptools_scm.version import get_local_dirty_tag

    def clean_scheme(version):
        return get_local_dirty_tag(version) if version.dirty else "+clean"

    return {"local_scheme": clean_scheme}


setup(
    name=NAME,
    use_scm_version=myversion,
    author=AUTHOR,
    author_email=EMAIL,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=URL,
    license="GPL",
    project_urls={
        "Bug Tracker": f"{URL}/issues",
    },
    python_requires=REQUIRES_PYTHON,
    packages=find_packages(exclude=("tests", "docs", "examples")),
    package_data={"grits": ["compounds/*"]},
    install_requires=REQUIRED,
    include_package_data=True,
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
)

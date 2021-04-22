from os import path
import pytest

import mbuild as mb


test_dir = path.dirname(__file__)


class BaseTest:
    @pytest.fixture
    def p3ht(self):
        p3ht = mb.load(path.join(test_dir, "assets/P3HT_16.mol2"))
        return p3ht

    @pytest.fixture
    def methane(self):
        methane =  mb.load("C", smiles=True)
        return methane

from os import path

import mbuild as mb
import pytest

from grits import CG_Compound

test_dir = path.dirname(__file__)


class BaseTest:
    def mapping(self, cg_comp, tmpdir):
        filename = tmpdir.mkdir("sub").join("mapping.txt")
        cg_comp.save_mapping(filename)
        return filename

    @pytest.fixture
    def p3ht(self):
        p3ht = mb.load(path.join(test_dir, "assets/P3HT_16.mol2"))
        return p3ht

    @pytest.fixture
    def methane(self):
        methane = mb.load("C", smiles=True)
        return methane

    @pytest.fixture
    def alkane(self):
        chain = mb.load("CCC" * 4, smiles=True)
        return chain

    @pytest.fixture
    def p3ht_mapping(self, p3ht):
        return mapping(p3ht)

    @pytest.fixture
    def methane_mapping(self, methane):
        return mapping(methane)

    @pytest.fixture
    def alkane_mapping(self, alkane):
        return mapping(alkane)

    @pytest.fixture
    def cg_p3ht(self, p3ht):
        cg_beads = {"_B": "c1sccc1", "_S": "CCC"}

        cg_p3ht = CG_Compound(p3ht, cg_beads)
        return cg_p3ht

    @pytest.fixture
    def cg_methane(self, methane):
        cg_beads = {"_A": "C"}

        cg_methane = CG_Compound(methane, cg_beads)
        return cg_methane

    @pytest.fixture
    def cg_alkane(self, alkane):
        cg_chain = CG_Compound(alkane, {"_A": "CCC"})
        return cg_chain

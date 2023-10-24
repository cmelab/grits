from os import path

import mbuild as mb
import pytest
import gsd.hoomd

from grits import CG_Compound

test_dir = path.dirname(__file__)


class BaseTest:
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
    def p3ht_mapping(self, cg_p3ht, tmpdir):
        filename = tmpdir.mkdir("sub").join("p3htmapping.json")
        cg_p3ht.save_mapping(filename)
        return filename

    @pytest.fixture
    def methane_mapping(self, cg_methane, tmpdir):
        filename = tmpdir.mkdir("sub").join("methanemapping.json")
        cg_methane.save_mapping(filename)
        return filename

    @pytest.fixture
    def alkane_mapping(self, cg_alkane):
        filename = tmpdir.mkdir("sub").join("alkanemapping.json")
        cg_alkane.save_mapping(filename)
        return filename

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

    @pytest.fixture
    def butane_gsd(self, tmpdir):
        molecule = mb.load("CCCC", smiles=True)
        filename = tmpdir.mkdir("sub").join("butane.gsd")
        molecule.save(filename)
        return gsd.hoomd.open(filename)
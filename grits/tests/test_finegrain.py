import pytest
from base_test import BaseTest
from mbuild import Compound

from grits import backmap


class Test_Backmap(BaseTest):
    def test_backmapnobonds(self, methane, cg_methane):
        fg_methane = backmap(cg_methane)

        assert isinstance(fg_methane, Compound)
        assert fg_methane.n_particles == methane.n_particles

    def test_backmapbonds(self, p3ht, cg_p3ht):
        fg_p3ht = backmap(cg_p3ht)

        assert isinstance(fg_p3ht, Compound)
        assert fg_p3ht.n_particles == p3ht.n_particles
        assert fg_p3ht.n_bonds == p3ht.n_bonds

    def test_alkane_chain(self):
        chain = mb.load("CCC"*4, smiles=True)
        cg_chain = grits.CG_Compound(chain, {"_A", "CCC"})
        fg_chain = backmap(cg_chain)
        assert fg_chain.n_bonds == chain.n_bonds
        assert fg_chain.n_particles == chain.n_particles


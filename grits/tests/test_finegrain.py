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

    def test_alkane(self, alkane, cg_alkane):
        fg_alkane = backmap(cg_alkane)

        assert fg_alkane.n_bonds == alkane.n_bonds
        assert fg_alkane.n_particles == alkane.n_particles

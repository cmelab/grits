import pytest

from mbuild import Compound

from base_test import BaseTest
from grits import backmap, CG_Compound


class Test_Backmap(BaseTest):
    def test_backmapnobonds(self, methane):
        cg_beads = [("_A", "C")]

        cg_methane = CG_Compound(methane, cg_beads)

        bead_dict = {"_A": {"smiles": "C", "anchors": []},}

        fg_methane = backmap(cg_methane, bead_dict)

        assert isinstance(fg_methane, Compound)
        assert fg_methane.n_particles == methane.n_particles

    def test_backmapbonds(self, p3ht):
        cg_beads = [("_B", "c1sccc1"), ("_S", "CCC")]

        cg_p3ht = CG_Compound(p3ht, cg_beads)

        bead_dict = {
            "_B": {"smiles": "c1sccc1", "anchors": [0, 2, 4]},
            "_S": {"smiles": "CCC", "anchors": [0, 2]},
        }

        bond_dict = {
            "_B_B": [(0, 2), (2, 0)],
            "_B_S": [(4, 0), (4, 2)],
            "_S_S": [(2, 0), (0, 2)],
        }

        fg_p3ht = backmap(cg_p3ht, bead_dict, bond_dict)

        assert isinstance(fg_p3ht, Compound)
        assert fg_p3ht.n_particles == p3ht.n_particles
        assert fg_p3ht.n_bonds == p3ht.n_bonds

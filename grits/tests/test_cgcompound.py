from base_test import BaseTest
from grits import CG_Compound


class Test_CGCompound(BaseTest):
    def test_init_methane(self, methane):
        cg_beads = [("_A", "C")]

        cg_methane = CG_Compound(methane, cg_beads)

        assert cg_methane.n_particles == 1
        assert isinstance(cg_methane, CG_Compound)

        types = set([i.name for i in cg_methane.particles()])
        assert "_A" in types
        assert len(types) == 1

    def test_init_p3ht(self, p3ht):
        cg_beads = [("_B", "c1sccc1"), ("_S", "CCC")]

        cg_p3ht = CG_Compound(p3ht, cg_beads)

        assert cg_p3ht.n_particles == 48
        assert isinstance(cg_p3ht, CG_Compound)

        types = set([i.name for i in cg_p3ht.particles()])
        assert "_B" in types
        assert "_S" in types
        assert len(types) == 2

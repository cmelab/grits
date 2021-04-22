from base_test import BaseTest


class Test_CGCompound(BaseTest):
    def test_init(self, p3ht)
        from grits import CG_Compound


        cg_beads = [("_B", "c1sccc1"), ("_S", "CCC")]

        cg_p3ht = CG_Compound(p3ht, cg_beads)

        assert cg_p3ht.n_particles == 48
        assert isinstance(cg_p3ht, CG_Compound)

        types = set([i.name for i in cg_p3ht.particles()])
        assert "_B" in types
        assert "_S" in types
        assert len(types) == 2

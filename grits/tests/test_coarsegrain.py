import tempfile
from os import path

import pytest
from base_test import BaseTest

from grits import CG_Compound, CG_System
from grits.utils import amber_dict

asset_dir = path.join(path.dirname(__file__), "assets")


class Test_CGCompound(BaseTest):
    def test_initmethane(self, methane):
        cg_beads = {"_A": "C"}

        cg_methane = CG_Compound(methane, cg_beads)

        assert cg_methane.n_particles == 1
        assert isinstance(cg_methane, CG_Compound)

        types = set([i.name for i in cg_methane.particles()])
        assert "_A" in types
        assert len(types) == 1

    def test_initp3ht(self, p3ht):
        cg_beads = {"_B": "c1sccc1", "_S": "CCC"}

        cg_p3ht = CG_Compound(p3ht, cg_beads)

        assert cg_p3ht.n_particles == 48
        assert isinstance(cg_p3ht, CG_Compound)

        types = set([i.name for i in cg_p3ht.particles()])
        assert "_B" in types
        assert "_S" in types
        assert len(types) == 2

    def test_initp3htoverlap(self, p3ht):
        cg_beads = {"_B": "c1sccc1", "_S": "CCC"}

        cg_p3ht = CG_Compound(p3ht, cg_beads, allow_overlap=True)

        assert cg_p3ht.n_particles == 48
        assert isinstance(cg_p3ht, CG_Compound)

        types = set([i.name for i in cg_p3ht.particles()])
        assert "_B" in types
        assert "_S" in types
        assert len(types) == 2

    def test_initmapp3ht(self, p3ht, p3ht_mapping):
        cg_p3ht = CG_Compound(p3ht, mapping=p3ht_mapping)

        assert cg_p3ht.n_particles == 48
        assert isinstance(cg_p3ht, CG_Compound)

        types = set([i.name for i in cg_p3ht.particles()])
        assert "_B" in types
        assert "_S" in types
        assert len(types) == 2

    def test_initmapmethane(self, methane, methane_mapping):
        cg_methane = CG_Compound(methane, mapping=methane_mapping)

        assert cg_methane.n_particles == 1
        assert isinstance(cg_methane, CG_Compound)

        types = set([i.name for i in cg_methane.particles()])
        assert "_A" in types
        assert len(types) == 1

    def test_notfoundsmarts(self, methane):
        cg_beads = {"_A": "CCC"}

        with pytest.warns(UserWarning):
            CG_Compound(methane, cg_beads)

    def test_atomsleftout(self, p3ht):
        cg_beads = {"_S": "CCC"}

        with pytest.warns(UserWarning):
            CG_Compound(p3ht, cg_beads)

    def test_badinit(self, p3ht):
        with pytest.raises(ValueError):
            CG_Compound(p3ht, beads="heck", mapping="this")
        with pytest.raises(ValueError):
            CG_Compound(p3ht)

    def test_savemapping(self, cg_methane):
        assert cg_methane.save_mapping() == "CG_Compound_mapping.json"

    def test_reprnoerror(self, cg_methane, cg_p3ht):
        str(cg_p3ht)
        str(cg_methane)


class Test_CGSystem(BaseTest):
    def test_p3ht(self, tmp_path):
        gsdfile = path.join(asset_dir, "p3ht.gsd")
        system = CG_System(
            gsdfile,
            beads={"_B": "c1cscc1", "_S": "CCC"},
            conversion_dict=amber_dict,
        )

        assert isinstance(system.mapping, dict)
        assert len(system.mapping["_B...c1cscc1"]) == 160

        cg_gsd = tmp_path / "cg-p3ht.gsd"
        system.save(cg_gsd)

        cg_json = tmp_path / "cg-p3ht.json"
        system.save_mapping(cg_json)

    def test_iticp3ht(self, tmp_path):
        gsdfile = path.join(asset_dir, "itic-p3ht.gsd")
        system = CG_System(
            gsdfile,
            beads={"_A": "c1c4c(cc2c1Cc3c2scc3)Cc5c4scc5", "_B": "c1cscc1"},
            conversion_dict=amber_dict,
        )

        assert isinstance(system.mapping, dict)
        assert len(system.mapping["_A...c1c4c(cc2c1Cc3c2scc3)Cc5c4scc5"]) == 10

        cg_gsd = tmp_path / "cg-itic-p3ht.gsd"
        system.save(cg_gsd)

        cg_json = tmp_path / "cg-itic-p3ht.json"
        system.save_mapping(cg_json)

import tempfile
from os import path

import gsd.hoomd
import numpy as np
import pytest
from base_test import BaseTest

from grits import CG_Compound, CG_System
from grits.utils import amber_dict

asset_dir = path.join(path.dirname(__file__), "assets")
propyl = 'CCC'
benzyl = 'c1ccccc1'
p3ht_backbone = 'c1sccc1'
itic_backbone = 'c1c2Cc3c4sccc4sc3c2cc5Cc6c7sccc7sc6c15'
itic_end = 'c1cccc2c1C(=O)C(=C)C2=C(C#N)C#N'


class Test_CGCompound(BaseTest):
    def test_initmethane(self, methane):
        cg_beads = {"_A": "C"}

        cg_methane = CG_Compound(methane, cg_beads)

        assert cg_methane.n_particles == 1
        assert isinstance(cg_methane, CG_Compound)

        types = set([i.name for i in cg_methane.particles()])
        assert "_A" in types
        assert len(types) == 1
        assert np.isclose(cg_methane.mass, 12.011)

    def test_initp3ht(self, p3ht):
        cg_beads = {"_B": p3ht_backbone, "_S": propyl}

        cg_p3ht = CG_Compound(p3ht, cg_beads)

        assert cg_p3ht.n_particles == 48
        assert isinstance(cg_p3ht, CG_Compound)

        types = set([i.name for i in cg_p3ht.particles()])
        assert "_B" in types
        assert "_S" in types
        assert len(types) == 2
        assert np.isclose(cg_p3ht[0].mass, 80.104)
        assert np.isclose(cg_p3ht[17].mass, 36.033)

    def test_initp3htoverlap(self, p3ht):
        cg_beads = {"_B": p3ht_backbone, "_S": propyl}

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
        cg_beads = {"_A": propyl}

        with pytest.warns(UserWarning):
            CG_Compound(methane, cg_beads)

    def test_atomsleftout(self, p3ht):
        cg_beads = {"_S": propyl}

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
    def test_raises(self):
        gsdfile = path.join(asset_dir, "p3ht.gsd")
        with pytest.raises(ValueError):
            CG_System(gsdfile)

    def test_p3ht(self, tmp_path):
        gsdfile = path.join(asset_dir, "p3ht.gsd")
        system = CG_System(
            gsdfile,
            beads={"_B": p3ht_backbone, "_S": propyl},
            conversion_dict=amber_dict,
        )

        assert isinstance(system.mapping, dict)
        assert len(system.mapping["_B...c1cscc1"]) == 160

        cg_gsd = tmp_path / "cg-p3ht.gsd"
        system.save(cg_gsd)
        with gsd.hoomd.open(cg_gsd) as f:
            snap = f[0]
            assert len(set(snap.particles.mass)) == 2
            assert np.isclose(snap.particles.mass[0], 2.49844)
            assert (
                len(snap.bonds.typeid) == len(snap.bonds.group) == snap.bonds.N
            )
            assert len(snap.bonds.types) == 3

        cg_json = tmp_path / "cg-p3ht.json"
        system.save_mapping(cg_json)

    def test_p3ht_noh(self, tmp_path):
        gsdfile = path.join(asset_dir, "p3ht-noH.gsd")
        system = CG_System(
            gsdfile,
            beads={"_B": p3ht_backbone, "_S": propyl},
            conversion_dict=amber_dict,
            add_hydrogens=True,
        )

        assert isinstance(system.mapping, dict)
        assert len(system.mapping["_B...c1cscc1"]) == 160

        cg_gsd = tmp_path / "cg-p3ht.gsd"
        system.save(cg_gsd)

        cg_json = tmp_path / "cg-p3ht.json"
        system.save_mapping(cg_json)

    def test_mass_scale(self, tmp_path):
        gsdfile = path.join(asset_dir, "itic-p3ht.gsd")
        with gsd.hoomd.open(gsdfile, "rb") as traj:
            init_mass = sum(traj[0].particles.mass)
        system = CG_System(
            gsdfile,
            beads={"_A": itic_backbone, "_B": itic_end, "_C": p3ht_backbone, "_S1": benzyl, "_S2": propyl},
            conversion_dict=amber_dict,
            mass_scale=2.0
        )

        cg_gsd = tmp_path / "cg-p3ht.gsd"
        system.save(cg_gsd)
        with gsd.hoomd.open(cg_gsd, "rb") as cg_traj:
            cg_mass = sum(cg_traj[0].particles.mass)
        assert np.allclose(cg_mass, init_mass*2, 1e-2)

    def test_iticp3ht(self, tmp_path):
        gsdfile = path.join(asset_dir, "itic-p3ht.gsd")
        system = CG_System(
            gsdfile,
            beads={"_A": itic_backbone, "_B": itic_end, "_C": p3ht_backbone, "_S1": benzyl, "_S2": propyl},
            conversion_dict=amber_dict,
        )

        assert isinstance(system.mapping, dict)
        assert len(system.mapping[f"_A...{itic_backbone}"]) == 10
        assert len(system.mapping[f"_B...{itic_end}"]) == 20

        cg_gsd = tmp_path / "cg-itic-p3ht.gsd"
        system.save(cg_gsd)

        cg_json = tmp_path / "cg-itic-p3ht.json"
        system.save_mapping(cg_json)

        map_system = CG_System(
            gsdfile,
            mapping=cg_json,
            conversion_dict=amber_dict,
        )

        assert isinstance(map_system.mapping, dict)
        assert (
            len(map_system.mapping[f"_A...{itic_backbone}"]) == 10,
            len(map_system.mapping[f"_B...{itic_end}"]) == 20
        )

        map_cg_gsd = tmp_path / "map-cg-itic-p3ht.gsd"
        system.save(map_cg_gsd)

        with gsd.hoomd.open(map_cg_gsd) as t_map, gsd.hoomd.open(cg_gsd) as t:
            map_s = t_map[0]
            s = t[0]
            assert s.particles.N == 170
            assert np.all(s.particles.position == map_s.particles.position)

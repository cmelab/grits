import tempfile
from os import path

import gsd.hoomd
import numpy as np
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
        assert np.isclose(cg_methane.mass, 16.043, atol=1e-5)

    def test_initp3ht(self, p3ht):
        cg_beads = {"_B": "c1sccc1", "_S": "CCC"}

        cg_p3ht = CG_Compound(p3ht, cg_beads)

        assert cg_p3ht.n_particles == 48
        assert isinstance(cg_p3ht, CG_Compound)

        types = set([i.name for i in cg_p3ht.particles()])
        assert "_B" in types
        assert "_S" in types
        assert len(types) == 2
        assert np.isclose(cg_p3ht[0].mass, 82.12, atol=1e-5)
        assert np.isclose(cg_p3ht[17].mass, 43.089, atol=1e-5)

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
    def test_raises(self):
        gsdfile = path.join(asset_dir, "benzene-aa.gsd")
        with pytest.raises(ValueError):
            CG_System(gsdfile)

    def test_pps(self, tmp_path):
        gsdfile = path.join(asset_dir, "pps-aa.gsd")
        system = CG_System(
            gsdfile,
            beads={"_B": "c1ccc(S)cc1"},
            conversion_dict=amber_dict,
        )

        assert isinstance(system.mapping, dict)
        assert len(system.mapping["_B...c1cc(S)ccc1"]) == 300 

        cg_gsd = tmp_path / "cg-pps.gsd"
        system.save(cg_gsd)
        with gsd.hoomd.open(cg_gsd) as f:
            snap = f[0]
            assert len(set(snap.particles.mass)) == 1 
            assert (
                len(snap.bonds.typeid) == len(snap.bonds.group) == snap.bonds.N
            )
            assert len(snap.bonds.types) == 1 

        cg_json = tmp_path / "cg-pps.json"
        system.save_mapping(cg_json)

    def test_pps_noH(self, tmp_path):
        gsdfile = path.join(asset_dir, "pps-ua.gsd")
        system = CG_System(
            gsdfile,
            beads={"_B": "c1ccc(S)cc1"},
            conversion_dict=amber_dict,
            add_hydrogens=True
        )

        assert isinstance(system.mapping, dict)
        assert len(system.mapping["_B...c1cc(S)ccc1"]) == 300 

        cg_gsd = tmp_path / "cg-pps.gsd"
        system.save(cg_gsd)
        with gsd.hoomd.open(cg_gsd) as f:
            snap = f[0]
            assert len(set(snap.particles.mass)) == 1 
            assert (
                len(snap.bonds.typeid) == len(snap.bonds.group) == snap.bonds.N
            )
            assert len(snap.bonds.types) == 1 

        cg_json = tmp_path / "cg-pps.json"
        system.save_mapping(cg_json)

    def test_mass_scale(self, tmp_path):
        gsdfile = path.join(asset_dir, "pps-aa.gsd")
        with gsd.hoomd.open(gsdfile, "r") as traj:
            init_mass = sum(traj[0].particles.mass)

        system = CG_System(
            gsdfile,
            beads={"_B": "c1ccc(S)cc1"},
            conversion_dict=amber_dict,
            mass_scale=2.0
        )
        cg_gsd = tmp_path / "cg-pps.gsd"
        system.save(cg_gsd)
        with gsd.hoomd.open(cg_gsd, "r") as cg_traj:
            cg_mass = sum(cg_traj[0].particles.mass)
        assert np.allclose(cg_mass, init_mass*2, 1e-2)

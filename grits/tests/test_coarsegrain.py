import tempfile
from os import path

import gsd.hoomd
import numpy as np
import pytest
from base_test import BaseTest

from grits import CG_Compound, CG_System
from grits.utils import amber_dict

asset_dir = path.join(path.dirname(__file__), "assets")
propyl = "CCC"
benzyl = "c1ccccc1"
p3ht_backbone = "c1sccc1"
itic_backbone = "c1c2Cc3c4sccc4sc3c2cc5Cc6c7sccc7sc6c15"
itic_end = "c1cccc2c1c(=o)c(=c*)c2=c(c#n)c#n"


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

    def test_initanisomethane(self, methane):
        cg_beads = {"_A": "C"}

        cg_methane = CG_Compound(methane, cg_beads, aniso_beads=True)

        assert cg_methane.n_particles == 1
        assert isinstance(cg_methane, CG_Compound)

        types = set([i.name for i in cg_methane.particles()])
        assert "_A" in types
        assert len(types) == 1
        assert np.isclose(cg_methane.mass, 16.043, atol=1e-5)
        # in methane we expect all default orientations
        for particle in cg_methane.particles():
            assert np.allclose(particle.orientation, np.array([0, 0, 0, 1]))

    def test_initp3ht(self, p3ht):
        cg_beads = {"_B": p3ht_backbone, "_S": propyl}

        cg_p3ht = CG_Compound(p3ht, cg_beads)

        assert cg_p3ht.n_particles == 48
        assert isinstance(cg_p3ht, CG_Compound)

        types = set([i.name for i in cg_p3ht.particles()])
        assert "_B" in types
        assert "_S" in types
        assert len(types) == 2
        assert np.isclose(cg_p3ht[0].mass, 82.12, atol=1e-5)
        assert np.isclose(cg_p3ht[17].mass, 43.089, atol=1e-5)

    def test_initanisop3ht(self, p3ht):
        cg_beads = {"_B": p3ht_backbone, "_S": propyl}

        cg_p3ht = CG_Compound(p3ht, cg_beads, aniso_beads=True)

        assert cg_p3ht.n_particles == 48
        assert isinstance(cg_p3ht, CG_Compound)

        types = set([i.name for i in cg_p3ht.particles()])
        assert "_B" in types
        assert "_S" in types
        assert len(types) == 2
        assert np.isclose(cg_p3ht[0].mass, 82.12, atol=1e-5)
        assert np.isclose(cg_p3ht[17].mass, 43.089, atol=1e-5)
        # larger beads should have non-default orientations
        for particle in cg_p3ht.particles():
            assert not np.allclose(particle.orientation, np.array([0, 0, 0, 1]))

    def test_initp3htoverlap(self, p3ht):
        cg_beads = {"_B": p3ht_backbone, "_S": propyl}

        cg_p3ht = CG_Compound(p3ht, cg_beads, allow_overlap=True)

        assert cg_p3ht.n_particles == 48
        assert isinstance(cg_p3ht, CG_Compound)

        types = set([i.name for i in cg_p3ht.particles()])
        assert "_B" in types
        assert "_S" in types
        assert len(types) == 2

    def test_initanisop3htoverlap(self, p3ht):
        cg_beads = {"_B": p3ht_backbone, "_S": propyl}

        cg_p3ht = CG_Compound(
            p3ht, cg_beads, allow_overlap=True, aniso_beads=True
        )

        assert cg_p3ht.n_particles == 48
        assert isinstance(cg_p3ht, CG_Compound)

        types = set([i.name for i in cg_p3ht.particles()])
        assert "_B" in types
        assert "_S" in types
        assert len(types) == 2
        for particle in cg_p3ht.particles():
            assert not np.allclose(particle.orientation, np.array([0, 0, 0, 1]))

    def test_initmapp3ht(self, p3ht, p3ht_mapping):
        cg_p3ht = CG_Compound(p3ht, mapping=p3ht_mapping)

        assert cg_p3ht.n_particles == 48
        assert isinstance(cg_p3ht, CG_Compound)

        types = set([i.name for i in cg_p3ht.particles()])
        assert "_B" in types
        assert "_S" in types
        assert len(types) == 2

    def test_initmapanisop3ht(self, p3ht, p3ht_mapping):
        cg_p3ht = CG_Compound(p3ht, mapping=p3ht_mapping, aniso_beads=True)

        assert cg_p3ht.n_particles == 48
        assert isinstance(cg_p3ht, CG_Compound)

        types = set([i.name for i in cg_p3ht.particles()])
        assert "_B" in types
        assert "_S" in types
        assert len(types) == 2
        for particle in cg_p3ht.particles():
            assert not np.allclose(particle.orientation, np.array([0, 0, 0, 1]))

    def test_initmapmethane(self, methane, methane_mapping):
        cg_methane = CG_Compound(methane, mapping=methane_mapping)

        assert cg_methane.n_particles == 1
        assert isinstance(cg_methane, CG_Compound)

        types = set([i.name for i in cg_methane.particles()])
        assert "_A" in types
        assert len(types) == 1

    def test_initmapanisomethane(self, methane, methane_mapping):
        cg_methane = CG_Compound(
            methane, mapping=methane_mapping, aniso_beads=True
        )

        assert cg_methane.n_particles == 1
        assert isinstance(cg_methane, CG_Compound)

        types = set([i.name for i in cg_methane.particles()])
        assert "_A" in types
        assert len(types) == 1
        for particle in cg_methane.particles():
            assert np.allclose(particle.orientation, np.array([0, 0, 0, 1]))

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
        gsdfile = path.join(asset_dir, "benzene-aa.gsd")
        with pytest.raises(ValueError):
            CG_System(gsdfile)

    def test_benzene(self, tmp_path):
        gsdfile = path.join(asset_dir, "benzene-aa.gsd")
        system = CG_System(
            gsdfile,
            beads={"_B": "c1ccccc1"},
            conversion_dict=amber_dict,
            mass_scale=12.011,
        )
        assert isinstance(system.mapping, dict)
        assert len(system.mapping["_B...c1ccccc1"]) == 20

        cg_gsd = tmp_path / "cg-benzene.gsd"
        system.save(cg_gsd)
        with gsd.hoomd.open(cg_gsd) as f:
            snap = f[0]
            assert len(set(snap.particles.mass)) == 1
            assert len(snap.bonds.typeid) == len(snap.bonds.group) == 0
            assert len(snap.bonds.types) == 0
            assert np.allclose(sum(snap.particles.mass), 20 * 78.11, atol=1e-1)

        cg_json = tmp_path / "cg-benzene.json"
        system.save_mapping(cg_json)

    def test_stride(self, tmp_path):
        gsdfile = path.join(asset_dir, "benzene-aa.gsd")
        system = CG_System(
            gsdfile,
            beads={"_B": "c1ccccc1"},
            conversion_dict=amber_dict,
            mass_scale=12.011,
        )
        cg_gsd = tmp_path / "cg-benzene.gsd"
        system.save(cg_gsdfile=cg_gsd, start=0, stop=-1, stride=2)
        with gsd.hoomd.open(cg_gsd) as f:
            assert len(f) == 3

    def test_anisobenzene(self, tmp_path):
        gsdfile = path.join(asset_dir, "benzene-aa.gsd")
        system = CG_System(
            gsdfile,
            beads={"_B": "c1ccccc1"},
            conversion_dict=amber_dict,
            aniso_beads=True,
            mass_scale=12.011,
        )
        assert isinstance(system.mapping, dict)
        assert len(system.mapping["_B...c1ccccc1"]) == 20

        cg_gsd = tmp_path / "cg-anisobenzene.gsd"
        system.save(cg_gsd)
        with gsd.hoomd.open(cg_gsd) as f:
            snap = f[0]
            assert len(set(snap.particles.mass)) == 1
            assert len(snap.bonds.typeid) == len(snap.bonds.group) == 0
            assert len(snap.bonds.types) == 0
            assert len(snap.particles.orientation) == 20
            assert np.allclose(sum(snap.particles.mass), 20 * 78.11, atol=1e-1)

        cg_json = tmp_path / "cg-anisobenzene.json"
        system.save_mapping(cg_json)

    def test_anisorotations(self, tmp_path):
        gsdfile = path.join(asset_dir, "four-pps-rotating.gsd")
        system = CG_System(
            gsdfile,
            beads={"_B": "c1ccc(S)cc1"},
            add_hydrogens=False,
            aniso_beads=True,
            mass_scale=12.011,
            allow_overlap=True,
        )
        cg_gsd = tmp_path / "cg-four-pps-rotating.gsd"
        system.save(cg_gsd)
        assert len(system.mapping["_B...c1ccc(S)cc1"]) == 4
        with gsd.hoomd.open(cg_gsd) as f:
            frame0 = f[0]
            frame1 = f[1]
        assert not np.allclose(
            frame0.particles.orientation, frame1.particles.orientation
        )

    def test_pps(self, tmp_path):
        gsdfile = path.join(asset_dir, "pps-aa.gsd")
        system = CG_System(
            gsdfile,
            beads={"_B": "c1ccc(S)cc1"},
            conversion_dict=amber_dict,
        )

        assert isinstance(system.mapping, dict)
        assert len(system.mapping["_B...c1ccc(S)cc1"]) == 225

        cg_gsd = tmp_path / "cg-pps.gsd"
        system.save(cg_gsd)
        with gsd.hoomd.open(cg_gsd) as f:
            snap = f[0]
            assert (
                len(snap.bonds.typeid) == len(snap.bonds.group) == snap.bonds.N
            )
            assert len(snap.bonds.types) == 1

            assert (
                len(snap.angles.typeid)
                == len(snap.angles.group)
                == snap.angles.N
            )
            assert snap.angles.N == 15 * 13
            assert snap.angles.types == ["_B-_B-_B"]

            assert (
                len(snap.dihedrals.typeid)
                == len(snap.dihedrals.group)
                == snap.dihedrals.N
            )
            assert snap.dihedrals.N == 15 * 12
            assert snap.dihedrals.types == ["_B-_B-_B-_B"]

        cg_json = tmp_path / "cg-pps.json"
        system.save_mapping(cg_json)

    def test_pps_noH(self, tmp_path):
        gsdfile = path.join(asset_dir, "pps-ua.gsd")
        system = CG_System(
            gsdfile,
            beads={"_B": "c1ccc([S,s])cc1"},
            conversion_dict=amber_dict,
            allow_overlap=False,
            add_hydrogens=True,
        )

        assert isinstance(system.mapping, dict)
        # this gsd has 15 15-mers of PPS -> expect 225 beads
        assert len(system.mapping["_B...c1ccc([S,s])cc1"]) == 225

        cg_gsd = tmp_path / "cg-pps.gsd"
        system.save(cg_gsd)
        with gsd.hoomd.open(cg_gsd) as f:
            snap = f[0]
            assert (
                len(snap.bonds.typeid) == len(snap.bonds.group) == snap.bonds.N
            )
            assert len(snap.bonds.types) == 1
            assert (
                len(snap.angles.typeid)
                == len(snap.angles.group)
                == snap.angles.N
            )
            assert snap.angles.N == 15 * 13
            assert snap.angles.types == ["_B-_B-_B"]

            assert (
                len(snap.dihedrals.typeid)
                == len(snap.dihedrals.group)
                == snap.dihedrals.N
            )
            assert snap.dihedrals.N == 15 * 12
            assert snap.dihedrals.types == ["_B-_B-_B-_B"]

        cg_json = tmp_path / "cg-pps.json"
        system.save_mapping(cg_json)

    def test_mass_scale(self, tmp_path):
        gsdfile = path.join(asset_dir, "benzene-aa.gsd")
        with gsd.hoomd.open(gsdfile, "r") as traj:
            init_mass = sum(traj[0].particles.mass)

        system = CG_System(
            gsdfile,
            beads={"_B": "c1ccccc1"},
            conversion_dict=amber_dict,
            mass_scale=12.011,
        )
        assert np.allclose(system._compounds[0].mass, 78.11, atol=1e-2)
        cg_gsd = tmp_path / "cg-benzene-scaled.gsd"
        system.save(cg_gsd)
        with gsd.hoomd.open(cg_gsd, "r") as cg_traj:
            cg_mass = sum(cg_traj[0].particles.mass)
        assert np.allclose(cg_mass, 20 * 78.11, atol=1)

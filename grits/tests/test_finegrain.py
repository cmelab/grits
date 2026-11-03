from base_test import BaseTest
from mbuild import Compound
import gsd.hoomd
from os import path
import pytest
from grits import backmap_compound, backmap_snapshot_to_compound
asset_dir = path.join(path.dirname(__file__), "assets")


class Test_Backmap(BaseTest):
    def test_backmapnobonds(self, methane, cg_methane):
        fg_methane = backmap_compound(cg_methane)

        assert isinstance(fg_methane, Compound)
        assert fg_methane.n_particles == methane.n_particles

    def test_backmapbonds(self, p3ht, cg_p3ht):
        fg_p3ht = backmap_compound(cg_p3ht)

        assert isinstance(fg_p3ht, Compound)
        assert fg_p3ht.n_particles == p3ht.n_particles
        assert fg_p3ht.n_bonds == p3ht.n_bonds

    def test_alkane(self, alkane, cg_alkane):
        fg_alkane = backmap_compound(cg_alkane)

        assert fg_alkane.n_bonds == alkane.n_bonds
        assert fg_alkane.n_particles == alkane.n_particles

    def test_backmap_snap_smiles(self,tmp_path):
        gsdfile = path.join(asset_dir, "cg-3mer.gsd")
        with gsd.hoomd.open(gsdfile,'r') as traj:
            snap = traj[-1]
        bead_mapping = {"_A": "C=CC1=CC=CC=C1"} # Mapping one A bead to 1 Polystyrene monomer
        head_indices = {"_A": [10]}
        tail_indices = {"_A": [9]}

        fg_comp = backmap_snapshot_to_compound(
            snapshot=snap,
            bead_mapping=bead_mapping,
            bond_head_index=head_indices,
            bond_tail_index=tail_indices,
            ref_distance=0.3438,
            energy_minimize=False
        )

        assert fg_comp.n_particles == 44
        assert fg_comp.n_bonds == 46

    def test_backmap_snap_library(self,tmp_path):
        gsdfile = path.join(asset_dir, "cg-3mer.gsd")
        with gsd.hoomd.open(gsdfile,'r') as traj:
            snap = traj[-1]
        fg_comp = backmap_snapshot_to_compound(
            snapshot=snap,
            library_key = 'polystyrene',
            ref_distance=0.3438,
            energy_minimize=False
        )

        assert fg_comp.n_particles == 44
        assert fg_comp.n_bonds == 46

    def test_backmap_snap_lists(self,tmp_path):
        gsdfile = path.join(asset_dir, "cg-3mer.gsd")
        with gsd.hoomd.open(gsdfile,'r') as traj:
            snap = traj[-1]
        bead_mapping = {"_A": "C[C@@H](C(=O)O)N"} # Mapping one A bead to 1 Polyalanine monomer
        head_indices = {"_A":[4,10]}
        tail_indices = {"_A":[12]}

        fg_comp = backmap_snapshot_to_compound(
            snapshot=snap,
            bead_mapping=bead_mapping,
            bond_head_index=head_indices,
            bond_tail_index=tail_indices,
            ref_distance=0.3438,
            energy_minimize=False
        )

        assert len(fg_comp.children) == snap.particles.N  # if each monomer is a sub-Compound

        expected_atoms_per_monomer = 11
        assert fg_comp.n_particles == snap.particles.N * expected_atoms_per_monomer

    def test_backmap_snap_not_int(self,tmp_path):
        gsdfile = path.join(asset_dir, "cg-3mer.gsd")
        with gsd.hoomd.open(gsdfile,'r') as traj:
            snap = traj[-1]
        bead_mapping = {"_A": "C[C@@H](C(=O)O)N"} # Mapping one A bead to 1 Polyalanine monomer
        head_indices = {"_A":[4.5,10]}
        tail_indices = {"_A":[12]}

        with pytest.raises(TypeError):
            fg_comp = backmap_snapshot_to_compound(
                snapshot=snap,
                bead_mapping=bead_mapping,
                bond_head_index=head_indices,
                bond_tail_index=tail_indices,
                ref_distance=0.3438,
                energy_minimize=False
            )

    def test_back_snap_end_hydrogens(self,tmp_path): 
        gsdfile = path.join(asset_dir, "cg-3mer.gsd")
        with gsd.hoomd.open(gsdfile,'r') as traj:
            snap = traj[-1]
        bead_mapping = {"_A": "CC"}
        head_indices = {"_A":[2]}
        tail_indices = {"_A":[5]}

        fg_comp = backmap_snapshot_to_compound(
            snapshot=snap,
            bead_mapping=bead_mapping,
            bond_head_index=head_indices,
            bond_tail_index=tail_indices,
            ref_distance=0.3,
            energy_minimize=False
        )

        assert fg_comp.n_particles == 20

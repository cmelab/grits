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

    def test_backmap_snap_smiles(snap):
        bead_mapping = {"A": "C=CC1=CC=CC=C1"} # Mapping one A bead to 1 Polystyrene monomer
        head_indices = {"A": [10]}
        tail_indices = {"A": [9]}
        
        fg_comp = backmap_snapshot_to_compound(
            snapshot=snap,
            bead_mapping=bead_mapping,
            bond_head_index=head_indices,
            bond_tail_index=tail_indices,
            ref_distance=0.3438,
            energy_minimize=False
        )
        
        assert fg_comp.particles == snap.particles
        assert fg_comp.bonds == snap.bonds

    def test_backmap_snap_library(snap):
        fg_comp = backmap_snapshot_to_compound(
            snapshot=snap,
            library_key = 'polystyrene'
            ref_distance=0.3438,
            energy_minimize=False
        )
        
        assert fg_comp.particles == snap.particles
        assert fg_comp.bonds == snap.bonds
        
    def test_backmap_snap_lists(snap):
        bead_mapping = {"A": "C[C@@H](C(=O)O)N"} # Mapping one A bead to 1 Polyalanine monomer
        head_indices = {"A":[4,10]}
        tail_indices = {"A":[12]}
        
        fg_comp = backmap_snapshot_to_compound(
            snapshot=snap,
            bead_mapping=bead_mapping,
            bond_head_index=head_indices,
            bond_tail_index=tail_indices,
            ref_distance=0.3438,
            energy_minimize=False
        )
        
        assert fg_comp.particles == snap.particles
        assert fg_comp.bonds == snap.bonds


    def test_backmap_snap_not_int(snap):
        bead_mapping = {"A": "C[C@@H](C(=O)O)N"} # Mapping one A bead to 1 Polyalanine monomer
        head_indices = {"A":[4.5,10]}
        tail_indices = {"A":[12]}
        
        fg_comp = backmap_snapshot_to_compound(
            snapshot=snap,
            bead_mapping=bead_mapping,
            bond_head_index=head_indices,
            bond_tail_index=tail_indices,
            ref_distance=0.3438,
            energy_minimize=False
        )
        pass #non-int should break code
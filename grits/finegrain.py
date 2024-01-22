"""GRiTS: Fine-graining tools."""
__all__ = ["backmap_compound", "backmap_gsd"]

import itertools as it
from collections import defaultdict

from mbuild import Compound, Particle, load
import mbuild as mb
import numpy as np
import gsd.hoomd

from grits.utils import align, get_hydrogen, get_index

def backmap_snapshot_to_compound(
        snapshot,
        bead_mapping,
        bond_head_index=dict(),
        bond_tail_index=dict()
):
    #TODO
    # assert all 3 dicts have the same keys
    cg_snap = snapshot
    fg_compound = mb.Compound()
    box = cg_snap.configuration.box
    pos_adjust = np.array([box[0] / 2, box[1] / 2, box[2] / 2])
    mb_box = mb.box.Box.from_lengths_angles(
            lengths=[box[0], box[1], box[2]],
            angles=[np.pi, np.pi, np.pi]
    )
    fg_compound.box = mb_box
    # Create atomistic compounds, remove hydrogens in the way of bonds
    compounds = dict() 
    anchor_dict = dict()
    for mapping in bead_mapping:
        comp = mb.load(bead_mapping[mapping], smiles=True)
        if bond_head_index and bond_tail_index:
            remove_hydrogens = [] # These will be removed
            anchor_particles = [] # Store this for making bonds later
            for i in [bond_tail_index[mapping], bond_head_index[mapping]]:
                for j, particle in enumerate(comp.particles()):
                    if j == i:
                        remove_hydrogens.append(particle)
                        anchor = [p for p in particle.direct_bonds()][0]
                        anchor_particles.append(anchor)
            for particle in remove_hydrogens:
                comp.remove(particle)
            # List of length 2 [tail particle index, head particle index]
            anchor_particle_indices = []
            for anchor in anchor_particles:
                for i, p in enumerate(comp.particles()):
                    if p == anchor:
                        anchor_particle_indices.append(i)
            anchor_dict[mapping] = tuple(anchor_particle_indices)

        compounds[mapping] = comp

    finished_beads = set()
    bead_to_comp_dict = dict()
    mb_compounds = []
    for group in cg_snap.bonds.group:
        for bead_index in group:
            if bead_index not in finished_beads:
                # add compoud to snapshot
                bead_type = cg_snap.particles.types[
                        cg_snap.particles.typeid[bead_index]
                ]
                bead_pos = cg_snap.particles.position[bead_index]
                comp = mb.clone(compounds[bead_type])
                comp.translate_to(bead_pos + pos_adjust)
                mb_compounds.append(comp)
                bead_to_comp_dict[bead_index] = comp
                finished_beads.add(bead_index)
                fg_compound.add(comp)

        tail_comp = bead_to_comp_dict[group[0]]
        tail_comp_particle = tail_comp[anchor_particle_indices[1]]
        head_comp = bead_to_comp_dict[group[1]]
        head_comp_particle = head_comp[anchor_particle_indices[0]]
        fg_compound.add_bond(particle_pair=[tail_comp_particle, head_comp_particle])
    return  fg_compound




def backmap_compound(cg_compound):
    """Backmap a fine-grained representation onto a coarse one.

    Creates a fine-grained compound from a coarse one using the attributes
    created during CG_Compound initialization.

    Parameters
    ----------
    cg_compound : CG_Compound
        Coarse-grained compound

    Returns
    -------
    :py:class:`mbuild.Compound`
        The atomistic structure mapped onto the coarse-grained one.
    """

    def fg_particles():
        """Set the particles of the fine-grained structure."""
        fine_grained = Compound()

        anchors = dict()
        for i, bead in enumerate(cg_compound):
            smiles = bead.smarts
            b = load(smiles, smiles=True)
            b.translate_to(bead.pos)
            anchors[i] = dict()
            if cg_compound.anchors is not None:
                for index in cg_compound.anchors[bead.name]:
                    anchors[i][index] = b[index]
            fine_grained.add(b, str(i))
        return fine_grained, anchors

    def fg_bonds():
        """Set the bonds for the fine-grained structure."""
        bonded_atoms = []
        remove_hs = []
        rotated = {k: False for k in anchors.keys()}
        for name, inds in cg_compound.bond_map:
            for ibead, jbead in cg_compound.bonds():
                names = [ibead.name, jbead.name]
                if "-".join(names) == name:
                    fi, fj = inds
                elif "-".join(names[::-1]) == name:
                    fj, fi = inds
                else:
                    continue

                i = get_index(cg_compound, ibead)
                j = get_index(cg_compound, jbead)
                try:
                    iatom = anchors[i].pop(fi)
                except KeyError:
                    fi = [x for x in inds if x in anchors[i]][0]
                    iatom = anchors[i].pop(fi)
                try:
                    jatom = anchors[j].pop(fj)
                except KeyError:
                    fj = [x for x in inds if x in anchors[j]][0]
                    jatom = anchors[j].pop(fj)

                hi = get_hydrogen(fine_grained, iatom)
                hj = get_hydrogen(fine_grained, jatom)
                # each part can be rotated
                if not rotated[i]:
                    # rotate
                    align(fine_grained[str(i)], hi, jbead)
                    rotated[i] = True
                if not rotated[j]:
                    # rotate
                    align(fine_grained[str(j)], hj, ibead)
                    rotated[j] = True

                fine_grained.add_bond((iatom, jatom))

                bonded_atoms += (iatom, jatom)
                remove_hs += (hi, hj)

        for atom in remove_hs:
            fine_grained.remove(atom)
        return fine_grained

    fine_grained, anchors = fg_particles()

    if cg_compound.bond_map is None:
        return fine_grained

    fine_grained = fg_bonds()
    return fine_grained

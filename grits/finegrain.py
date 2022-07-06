"""GRiTS: Fine-graining tools."""
__all__ = ["backmap"]

import itertools as it
import tempfile
from collections import defaultdict

import gsd.hoomd
import numpy as np
from ele import element_from_atomic_number
from mbuild import Compound, Particle, load
from openbabel import openbabel, pybel

from grits.utils import (
    align,
    comp_from_snapshot,
    get_hydrogen,
    get_index,
    snap_molecules,
)


def _add_hydrogens(snap, conversion_dict=None, scale=1.0):
    """Add hydrogens to a united atom gsd snapshot.

    Parameters
    ----------
    snap : gsd.hoomd.Snapshot
        The united-atom gsd snapshot
    conversion_dict : dict, default None
        Dictionary to map particle types to their element.
    scale : float, default 1.0
        Factor by which to scale length values.

    Returns
    -------
    gsd.hoomd.Snapshot
    """
    # Use the conversion dictionary to map particle type to element symbol
    if conversion_dict is not None:
        snap.particles.types = [
            conversion_dict[i].symbol for i in snap.particles.types
        ]
    # Break apart the snapshot into separate molecules
    molecules = snap_molecules(snap)
    mol_inds = []
    for i in range(max(molecules) + 1):
        mol_inds.append(np.where(molecules == i)[0])

    types = []
    coords = []
    bonds = []
    # Read each molecule into a compound
    # Then convert to pybel to add hydrogens
    for inds in mol_inds:
        l = len(inds)
        compound = comp_from_snapshot(
            snap, inds, scale=scale, shift_coords=False
        )
        mol = compound.to_pybel()
        mol.OBMol.PerceiveBondOrders()

        # Add hydrogens
        # This is a goofy work around necessary for the aromaticity
        # to be set correctly.
        with tempfile.NamedTemporaryFile() as f:
            mol.write(format="mol2", filename=f.name, overwrite=True)
            mol = list(pybel.readfile("mol2", f.name))[0]

        mol.addh()
        for atom in mol.atoms:
            coords.append(np.array(atom.coords) / 10)
            types.append(element_from_atomic_number(atom.atomicnum).symbol)
        for bond in openbabel.OBMolBondIter(mol.OBMol):
            bonds.append([bond.GetBeginAtomIdx() - 1, bond.GetEndAtomIdx() - 1])
    alltypes = list(set(types))
    typeids = [alltypes.index(i) for i in types]
    new_snap = gsd.hoomd.Snapshot()
    new_snap.configuration.box = snap.configuration.box
    new_snap.particles.N = len(types)
    new_snap.particles.types = alltypes
    new_snap.particles.typeid = typeids
    new_snap.particles.position = np.array(coords)
    new_snap.bonds.N = len(bonds)
    new_snap.bonds.group = np.array(bonds)
    new_snap.validate()
    return new_snap


def backmap(cg_compound):
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

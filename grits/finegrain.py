"""GRiTS: Fine-graining tools."""
__all__ = ["backmap"]

import itertools as it
from collections import defaultdict

from mbuild import Compound, Particle, load

from grits.utils import align, get_hydrogen, get_index


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

"""GRiTS: Fine-graining tools."""
from collections import defaultdict

from mbuild import Compound, Particle, load

from grits.utils import distance, get_index, remove_hydrogen


def backmap(cg_compound):
    """Backmap a fine-grained representation onto a coarse one.

    Creates a fine-grained compound from a coarse one given dictionaries
    specifying the bead and how to place bonds.

    Parameters
    ----------
    cg_compound: CG_Compound
        coarse-grained compound

    Returns
    -------
    mbuild.Compound
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
            for index in cg_compound.anchors[bead.name]:
                anchors[i][index] = b[index]
            fine_grained.add(b)
        return fine_grained, anchors

    def fg_bonds():
        """Set the bonds for the fine-grained structure."""
        bonded_atoms = []
        for name, inds in cg_compound.bond_map:
            for ibead, jbead in cg_compound.bonds():
                names = [ibead.name, jbead.name]
                if "-".join(names) == name:
                    pass
                elif "-".join(names[::-1]) == name:
                    inds = inds[::-1]
                    print(inds)
                else:
                    continue

                i = get_index(cg_compound, ibead)
                j = get_index(cg_compound, jbead)
                print(name, inds)
                iatom = anchors[i][inds[0]]
                jatom = anchors[j][inds[1]]

                if any(x in bonded_atoms for x in [iatom, jatom]):
                    continue
                fine_grained.add_bond((iatom, jatom))

                bonded_atoms.append(iatom)
                bonded_atoms.append(jatom)

        for atom in bonded_atoms:
            remove_hydrogen(fine_grained, atom)
        return fine_grained

    fine_grained, anchors = fg_particles()

    if cg_compound.bond_map is None:
        return fine_grained

    fine_grained = fg_bonds()
    return fine_grained

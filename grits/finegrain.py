"""GRiTS: Fine-graining tools."""
import itertools as it
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
                    xinds = inds
                elif "-".join(names[::-1]) == name:
                    xinds = inds[::-1]
                else:
                    continue

                # if the bonds is between two same type beads, we can try
                # bonding to other anchor sites.
                if ibead.name == jbead.name:
                    print("same")
                    # We'll choose based on distance
                    # start with a crazy big distance, so at least ONE pair
                    # will be better than it.
                    mindist = max(cg_compound.boundingbox.lengths)
                    for fi, fj in it.product(xinds, repeat=2):
                        print("\tnew: ", fi, fj)
                        iatom = anchors[i][fi]
                        jatom = anchors[j][fj]
                        if any(x in bonded_atoms for x in [iatom, jatom]):
                            # assume only one bond from the CG translates
                            # to the FG structure
                            print("\tskipping: ", fi, fj)
                            continue
                        dist = distance(iatom.pos, jatom.pos)
                        if dist < mindist:
                            print("\tnew best: ", fi, fj)
                            fi_best = fi
                            fj_best = fj
                            mindist = dist
                        print("\t\tbest: ", fi_best, fj_best)
                    print("final: ", fi_best, fj_best)
                else:
                    fi_best, fj_best = xinds

                i = get_index(cg_compound, ibead)
                j = get_index(cg_compound, jbead)
                iatom = anchors[i][fi_best]
                jatom = anchors[j][fj_best]

                fine_grained.add_bond((iatom, jatom))

                bonded_atoms += (iatom, jatom)

        for atom in bonded_atoms:
            remove_hydrogen(fine_grained, atom)
        return fine_grained

    fine_grained, anchors = fg_particles()

    if cg_compound.bond_map is None:
        return fine_grained

    fine_grained = fg_bonds()
    return fine_grained

from collections import defaultdict

from mbuild import load, Particle, Compound

from grits.utils import distance, get_index, remove_hydrogen


def backmap(cg_compound, bead_dict, bond_dict=None):
    """
    Creates a fine-grained compound from a coarse one given dictionaries
    specifying the bead and how to place bonds.

    Parameters
    ----------
    cg_compound: mbuild.Compound, coarse grained compound
    bead_dict: dictionary of dictionaries, specifies what SMILES string
               and bond anchors to use for each bead type
               For example:
                 bead_dict = {
                 "_B": {
                     "smiles": "c1sccc1",
                     "anchors": [0,2,4],
                     "posres": 1
                     },
                 }
               specifies that coarse grain bead "_B" should be replaced
               with the fine-grain structure represented by the SMILES string
               "c1sccc1", should form bonds to other fine-grained beads
               from atoms 0, 2, and 4, and should have a position restraint
               attached to atom 1 (optional).
    bond_dict: dictionary of list of tuples, specifies what fine-grain bond
               should replace the bonds in the coarse structure.
               For example:
                bond_dict = {
                    "_B_B": [(0,2),(2,0)],
                }
               specifies that the bond between two "_B" beads should happen
               in their fine-grain replacement between the 0th and 2nd or
               the 2nd and 0th atoms

    Returns
    -------
    mbuild.Compound
    """
    fine_grained, anchors = fg_particles(cg_compound, bead_dict)

    if bond_dict is None:
        return fine_grained

    fine_grained = fg_bonds(cg_compound, bond_dict, anchors, fine_grained)
    return fine_grained


def fg_particles(cg_compound, bead_dict):
    fine_grained = Compound()

    anchors = dict()
    for i,bead in enumerate(cg_compound):
        smiles = bead_dict[bead.name]["smiles"]
        b = load(smiles, smiles=True)
        b.translate_to(bead.pos)
        anchors[i] = dict()
        for index in bead_dict[bead.name]["anchors"]:
            anchors[i][index] = b[index]
        try:
            posres_ind = bead_dict[bead.name]["posres"]
            posres = Particle(name="X", pos=bead.pos)
            b.add(posres)
            b.add_bond((posres,b[posres_ind]))
        except KeyError:
            pass
        fine_grained.add(b)
    return fine_grained, anchors


def fg_bonds(cg_compound, bond_dict, anchors, fine_grained):
    bonded_atoms = []
    for ibead,jbead in cg_compound.bonds():
        i = get_index(cg_compound, ibead)
        j = get_index(cg_compound, jbead)
        names = [ibead.name,jbead.name]
        bondname = "".join(names)
        try:
            bonds = bond_dict[bondname]
        except KeyError:
            try:
                bondname = "".join(names[::-1])
                bonds = [(j,i) for (i,j) in bond_dict[bondname]]
            except KeyError:
                raise KeyError(f"{bondname} not defined in bond dictionary.")
        # choose a starting distance that is way too big
        mindist = max(cg_compound.boundingbox.lengths)
        for fi,fj in bonds:
            iatom = anchors[i][fi]
            jatom = anchors[j][fj]
            if (iatom in bonded_atoms) or (jatom in bonded_atoms):
                # assume only one bond from the CG translates
                # to the FG structure
                continue
            dist = distance(iatom.pos,jatom.pos)
            if dist < mindist:
                fi_best = fi
                fj_best = fj
                mindist = dist
        iatom = anchors[i][fi_best]
        jatom = anchors[j][fj_best]
        fine_grained.add_bond((iatom, jatom))

        bonded_atoms.append(iatom)
        bonded_atoms.append(jatom)

    for atom in bonded_atoms:
        remove_hydrogen(fine_grained,atom)
    return fine_grained

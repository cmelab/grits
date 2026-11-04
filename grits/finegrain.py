"""GRiTS: Fine-graining tools."""

import mbuild as mb
import numpy as np
from cmeutils.geometry import angle_between_vectors
from mbuild import Compound, load

from grits.utils import (
    align,
    get_hydrogen,
    get_index,
    reactant_dict,
)


def _build_template(
    smiles, tail_indices, head_indices, remove_tail, remove_head
):
    """Load a monomer and remove reacting atoms only on the requested sides.

    Returns the compound and (tail_anchor_index, head_anchor_index).
    """
    comp = mb.load(smiles, smiles=True)
    particles = list(comp.particles())
    # Anchor = atom bonded to the first listed (leaving) atom on each side
    tail_anchor = [p for p in particles[tail_indices[0]].direct_bonds()][0]
    head_anchor = [p for p in particles[head_indices[0]].direct_bonds()][0]
    remove_atoms = []
    if remove_tail:
        remove_atoms += [particles[i] for i in tail_indices]
    if remove_head:
        remove_atoms += [particles[i] for i in head_indices]
    for particle in remove_atoms:
        comp.remove(particle)
    particles = list(comp.particles())
    return comp, (particles.index(tail_anchor), particles.index(head_anchor))


def _minimum_image(vec, box):
    """Apply the minimum-image convention to a displacement in a HOOMD box."""
    Lx, Ly, Lz, xy, xz, yz = box
    box_matrix = np.array(
        [[Lx, xy * Ly, xz * Lz], [0, Ly, yz * Lz], [0, 0, Lz]]
    )
    frac = np.linalg.solve(box_matrix, vec)
    return box_matrix @ (frac - np.round(frac))


def _orient_bonds(bonds, n_particles):
    """Direct each bond along its chain, independent of stored pair order.

    Each chain is walked from its lowest-index end bead, so the result only
    depends on connectivity.
    """
    neighbors = {i: [] for i in range(n_particles)}
    for i, j in bonds:
        neighbors[int(i)].append(int(j))
        neighbors[int(j)].append(int(i))
    # Chain ends first so each walk starts at a terminal bead
    starts = sorted(neighbors, key=lambda i: (len(neighbors[i]) != 1, i))
    visited = set()
    oriented = []
    for start in starts:
        if start in visited:
            continue
        visited.add(start)
        stack = [start]
        while stack:
            i = stack.pop()
            for j in neighbors[i]:
                if j not in visited:
                    visited.add(j)
                    oriented.append((i, j))
                    stack.append(j)
    if len(oriented) != len(bonds):
        raise ValueError("Ring structures are not supported.")
    return np.array(oriented, dtype=int).reshape(-1, 2)


def backmap_snapshot_to_compound(
    snapshot,
    bead_mapping=None,
    bond_head_index=None,
    bond_tail_index=None,
    library_key=None,
    ref_distance=None,
    energy_minimize=False,
):
    """Backmap a fine-grained snapshot onto a coarse one.

    Creates a fine-grained compound from a coarse one using the given parameters.

    Parameters
    ----------
    snapshot : gsd.hoomd.Frame
        one frame of gsd trajectory

    bead_mapping : string, default None
        SMILES string of fine-grain monomer

    bond_head_index : list of int, default None
        index of polymer reaction, choose head to orient chain

    bond_tail_index : list of int, default None
        index of polymer reaction, choose tail to orient chain

    library_key : string, default None
        key to dictionary, reactant_dict, with known bead_mapping, bond_head_index, and bond_tail_index

    ref_distance : float, default None
        distance between monomers

    energy_minimize : bool, default False
        Option to run a mbuild energy minimization.

    Returns
    -------
    :py:class:`mbuild.Compound`
        The atomistic structure mapped onto the coarse-grained one.
    """

    provided = (
        bead_mapping is not None,
        bond_head_index is not None,
        bond_tail_index is not None,
    )
    if (library_key is None and not all(provided)) or (
        library_key is not None and any(provided)
    ):
        raise ValueError(
            "Provide all three mapping dictionaries or a library key, not both."
        )
    if library_key is not None:
        bead_mapping = reactant_dict[library_key]["smiles"]
        bond_tail_index = reactant_dict[library_key]["tail_indices"]
        bond_head_index = reactant_dict[library_key]["head_indices"]
    if not ref_distance:
        ref_distance = 1
    cg_snap = snapshot
    fg_compound = mb.Compound()
    box = cg_snap.configuration.box * ref_distance
    pos_adjust = np.array([box[0] / 2, box[1] / 2, box[2] / 2])
    mb_box = mb.box.Box.from_lengths_tilt_factors(
        lengths=box[:3], tilt_factors=box[3:]
    )
    fg_compound.box = mb_box

    # Orient bonds along each chain so head/tail don't depend on pair order
    bond_groups = _orient_bonds(
        cg_snap.bonds.group if cg_snap.bonds.N else [], cg_snap.particles.N
    )
    # Beads that bond via their head side (group[0]) or tail side (group[1])
    bonds_on_head = set(int(i) for i in bond_groups[:, 0])
    bonds_on_tail = set(int(i) for i in bond_groups[:, 1])

    # Minimum-image bond vectors; each bead is oriented along its first bond
    bead_to_bond_vec = dict()
    image_shifts = []
    for group in bond_groups:
        raw_vec = (
            cg_snap.particles.position[group[1]]
            - cg_snap.particles.position[group[0]]
        )
        cg_bond_vec = _minimum_image(raw_vec, cg_snap.configuration.box)
        image_shifts.append((cg_bond_vec - raw_vec) * ref_distance)
        cg_bond_vec = cg_bond_vec / np.linalg.norm(cg_bond_vec)
        for bead_index in group:
            bead_to_bond_vec.setdefault(int(bead_index), cg_bond_vec)

    compounds = (
        dict()
    )  # (bead_type, remove_tail, remove_head) -> (comp, anchors)
    bead_to_comp_dict = dict()
    bead_to_anchors = dict()
    mb_compounds = []
    # Create a compound for every particle, including unbonded ones
    for bead_index in range(cg_snap.particles.N):
        bead_type = cg_snap.particles.types[
            cg_snap.particles.typeid[bead_index]
        ]
        key = (
            bead_type,
            bead_index in bonds_on_tail,  # remove tail side?
            bead_index in bonds_on_head,  # remove head side?
        )
        if key not in compounds:
            compounds[key] = _build_template(
                bead_mapping[bead_type],
                bond_tail_index[bead_type],
                bond_head_index[bead_type],
                remove_tail=key[1],
                remove_head=key[2],
            )
        template, anchors = compounds[key]
        bead_pos = cg_snap.particles.position[bead_index] * ref_distance
        comp = mb.clone(template)
        if bead_index in bead_to_bond_vec:
            cg_bond_vec = bead_to_bond_vec[bead_index]
            tail_pos = comp[anchors[1]].xyz[0]
            head_pos = comp[anchors[0]].xyz[0]
            head_tail_vec = tail_pos - head_pos
            head_tail_vec = head_tail_vec / np.linalg.norm(head_tail_vec)
            normal_vec = np.cross(head_tail_vec, cg_bond_vec)
            if np.linalg.norm(normal_vec) < 1e-8:
                # (Anti-)parallel: rotate around any perpendicular axis
                normal_vec = np.cross(head_tail_vec, [1, 0, 0])
                if np.linalg.norm(normal_vec) < 1e-8:
                    normal_vec = np.cross(head_tail_vec, [0, 1, 0])
            angle = angle_between_vectors(
                head_tail_vec, cg_bond_vec, degrees=False
            )
            comp.rotate(around=normal_vec, theta=angle)
        comp.translate_to(bead_pos + pos_adjust)
        mb_compounds.append(comp)
        bead_to_comp_dict[bead_index] = comp
        bead_to_anchors[bead_index] = anchors
        fg_compound.add(comp)

    n_bonds = len(bond_groups)
    print_every = max(1, n_bonds // 20)  # ~20 updates total
    for bond_count, (group, image_shift) in enumerate(
        zip(bond_groups, image_shifts), start=1
    ):
        if bond_count % print_every == 0 or bond_count == n_bonds:
            print(
                f"Backmapping bond {bond_count}/{n_bonds} "
                f"({100 * bond_count / n_bonds:.0f}%)",
                flush=True,
            )
        tail_anchor_idx = bead_to_anchors[group[0]][1]
        head_anchor_idx = bead_to_anchors[group[1]][0]
        tail_comp = bead_to_comp_dict[group[0]]
        tail_comp_particle = tail_comp[tail_anchor_idx]
        head_comp = bead_to_comp_dict[group[1]]
        head_comp_particle = head_comp[head_anchor_idx]
        fg_compound.add_bond(
            particle_pair=[tail_comp_particle, head_comp_particle]
        )
        if energy_minimize:
            temp_head = mb.clone(head_comp)
            temp_tail = mb.clone(tail_comp)
            # Bring head next to tail if the bond crosses the box boundary
            temp_head.translate(image_shift)
            temp_comp = mb.Compound(subcompounds=[temp_tail, temp_head])
            tail_comp_particle = temp_tail[tail_anchor_idx]
            head_comp_particle = temp_head[head_anchor_idx]
            temp_comp.add_bond(
                particle_pair=[tail_comp_particle, head_comp_particle]
            )
            print("Running energy minimization")
            temp_comp.energy_minimize(steps=500)
            temp_tail = temp_comp.children[0]
            temp_head = temp_comp.children[1]
            tail_comp.xyz = temp_tail.xyz
            head_comp.xyz = temp_head.xyz - image_shift
    return fg_compound


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

        anchors = {}
        for i, bead in enumerate(cg_compound):
            smiles = bead.smarts
            b = load(smiles, smiles=True)
            b.translate_to(bead.pos)
            anchors[i] = {}
            if cg_compound.anchors is not None:
                for index in cg_compound.anchors[bead.name]:
                    anchors[i][index] = b[index]
            fine_grained.add(b, str(i))
        return fine_grained, anchors

    def fg_bonds():
        """Set the bonds for the fine-grained structure."""
        bonded_atoms = []
        remove_hs = []
        rotated = {k: False for k in anchors}
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
                    fi = next(x for x in inds if x in anchors[i])
                    iatom = anchors[i].pop(fi)
                try:
                    jatom = anchors[j].pop(fj)
                except KeyError:
                    fj = next(x for x in inds if x in anchors[j])
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

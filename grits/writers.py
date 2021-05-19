import gsd
import gsd.hoomd
import numpy as np


def write_snapshot(beads):
    """
    beads : iterable, required
        An iterable of any of the structure classes in cg_gsd.py
    """
    all_types = []
    all_pairs = []
    pair_groups = []
    all_angles = []
    all_pos = []
    box = None

    for idx, bead in enumerate(beads):
        if box is None:
            box = bead.system.box
        all_types.append(bead.name)
        all_pos.append(bead.center)
        try:
            if bead.parent == beads[idx + 1].parent:
                pair = list(set([bead.name, beads[idx + 1].name]))
                if len(pair) == 1:
                    pair *= 2
                all_pairs.append(f"{pair[0]}-{pair[1]}")
                pair_groups.append([idx, idx + 1])
                pair_groups.append([idx, idx + 1])
        except IndexError:
            pass

    types = list(set(all_types))
    pairs = list(set(all_pairs))
    type_ids = [np.where(np.array(types) == i)[0][0] for i in all_types]
    pair_ids = [np.where(np.array(pairs) == i)[0][0] for i in all_pairs]

    s = gsd.hoomd.Snapshot()
    s.particles.N = len(all_types)
    s.bonds.N = len(all_pairs)
    s.bonds.M = 2
    s.angles.N = len(all_angles)
    s.particles.types = types
    s.particles.typeids = np.array(type_ids)
    s.particles.position = np.array(all_pos)
    s.bonds.types = pairs
    s.bonds.typeid = np.array(pair_ids)
    s.bonds.group = np.vstack(pair_groups)
    s.configuration.box = box
    return s

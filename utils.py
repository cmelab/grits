import re
from collections import defaultdict

import freud
import gsd
import gsd.hoomd
import gsd.pygsd
import mbuild as mb
import numpy as np
from openbabel import pybel

from cg_compound import CG_Compound


def mb_to_freud_box(box):
    """
    Convert an mbuild box object (lengths, angles) to a freud box object (lengths, tilts)
    These sites were useful reference to calculate the box tilts from the angles:
    http://gisaxs.com/index.php/Unit_cell
    https://hoomd-blue.readthedocs.io/en/stable/box.html

    Parameters
    ----------
    box : mbuild.box.Box()

    Returns
    -------
    freud.box.Box()
    """
    Lx = box.lengths[0]
    Ly = box.lengths[1]
    Lz = box.lengths[2]
    alpha = box.angles[0]
    beta = box.angles[1]
    gamma = box.angles[2]

    frac = (
        np.cos(np.radians(alpha)) - np.cos(np.radians(beta)) * np.cos(np.radians(gamma))
    ) / np.sin(np.radians(gamma))

    c = np.sqrt(1 - np.cos(np.radians(beta)) ** 2 - frac ** 2)

    xy = np.cos(np.radians(gamma)) / np.sin(np.radians(gamma))
    xz = frac / c
    yz = np.cos(np.radians(beta)) / c

    box_list = list(box.maxs) + [xy, yz, xz]
    return freud.box.Box(*box_list)


def bin_distribution(vals, nbins, start=None, stop=None):
    """
    Calculates a distribution given an array of data

    Parameters
    ----------
    vals : np.ndarry (N,), values over which to calculate the distribution
    start : float, value to start bins (default min(bonds_dict[bond]))
    stop : float, value to stop bins (default max(bonds_dict[bond]))
    step : float, step size between bins (default (stop-start)/30)

    Returns
    -------
    np.ndarray (nbins,2), where the first column is the mean value of the bin and
    the second column is number of values which fell into that bin
    """
    if start == None:
        start = min(vals)
    if stop == None:
        stop = max(vals)
    step = (stop-start)/nbins

    bins = [i for i in np.arange(start, stop, step)]
    dist = np.empty([len(bins)-1,2])
    for i, length in enumerate(bins[1:]):
        in_bin = [b for b in vals if b > bins[i] and b < bins[i+1]]
        dist[i,1] = len(in_bin)
        dist[i,0] = np.mean((bins[i],bins[i+1]))
    return dist


def autocorr1D(array):
    """
    Takes in a linear numpy array, performs autocorrelation
    function and returns normalized array with half the length
    of the input
    """
    ft = np.fft.rfft(array - np.average(array))
    acorr = np.fft.irfft(ft * np.conjugate(ft)) / (len(array) * np.var(array))
    return acorr[0 : len(acorr) // 2]


def get_decorr(acorr):
    """
    Returns the decorrelation time of the autocorrelation, a 1D numpy array
    """
    return np.argmin(acorr > 0)


def error_analysis(data):
    """
    Returns the standard and relative error given a dataset in a 1D numpy array
    """
    serr = np.std(data) / np.sqrt(len(data))
    rel_err = np.abs(100 * serr / np.average(data))
    return (serr, rel_err)


def get_angle(a, b, c):
    """
    Calculates the angle between three points a-b-c

    Parameters
    ----------
    a,b,c : np.ndarrays, positions of points a, b, and c

    Returns
    -------
    float, angle in radians
    """
    ba = a - b
    bc = c - b

    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.arccos(cos)


def distance(pos1, pos2):
    """
    Calculates euclidean distance between two points.

    Parameters
    ----------
    pos1, pos2 : ((3,) numpy.ndarray), xyz coordinates
        (2D also works)

    Returns
    -------
    float distance
    """
    return np.linalg.norm(pos1 - pos2)


def v_distance(pos_array, pos2):
    """
    Calculates euclidean distances between all points in pos_array and pos2.

    Parameters
    ----------
    pos_array : ((N,3) numpy.ndarray), array of coordinates
    pos2 : ((3,) numpy.ndarray), xyz coordinate
        (2D also works)

    Returns
    -------
    (N,) numpy.ndarray of distances
    """
    return np.linalg.norm(pos_array - pos2, axis=1)


def get_molecules(snapshot):
    """
    Creates list of sets of connected atom indices

    This code adapted from Matias Thayer's:
    https://stackoverflow.com/questions/10301000/python-connected-components

    Parameters
    ----------
    snapshot : gsd.hoomd.Snapshot

    Returns
    -------
    list of sets of connected atom indices
    """

    def _snap_bond_graph(snapshot):
        """
        Given a snapshot from a trajectory create a graph of the bonds

        get_molecules(gsd.hoomd.Snapshot) --> dict of sets
        """
        bond_array = snapshot.bonds.group
        bond_graph = defaultdict(set)
        for row in bond_array:
            bond_graph[row[0]].add(row[1])
            bond_graph[row[1]].add(row[0])
        return bond_graph

    def _get_connected_group(node, already_seen):
        """
        This code adapted from Matias Thayer's:
        https://stackoverflow.com/questions/10301000/python-connected-components
        """

        result = set()
        nodes = set([node])
        while nodes:
            node = nodes.pop()
            already_seen.add(node)
            nodes.update(graph[node] - already_seen)
            result.add(node)
        return result, already_seen

    graph = _snap_bond_graph(snapshot)

    already_seen = set()
    result = []
    for node in graph:
        if node not in already_seen:
            connected_group, already_seen = _get_connected_group(node, already_seen)
            result.append(connected_group)
    return result


def gsd_rdf(gsdfile, A_name, B_name, start=0, stop=None, rmax=None, bins=50):
    """
    This function calculates the radial distribution function given
    a gsd file and the names of the particles. By default it will calculate
    the rdf for the entire the trajectory.

    Parameters
    ----------
    gsdfile : str, filename of the gsd trajectory
    A_name, B_name : str, name(s) of particles between which to calculate the rdf
        (found in gsd.hoomd.Snapshot.particles.types)
    start : int, which frame to start accumulating the rdf (default 0)
        (negative numbers index from the end)
    stop : int, which frame to stop accumulating the rdf (default None)
        If none is given, the function will default to the last frame.
    rmax : float, maximum radius to consider. (default None)
        If none is given, it'll be the minimum box length / 4
    rdf : freud.density.RDF, if provided, this function will accumulate an average rdf,
        otherwise it will provide the rdf only for the given compound. (default None)
    bins : int, number of bins to use when calculating the distribution.

    Returns
    -------
    freud.density.RDF
    """
    f = gsd.pygsd.GSDFile(open(gsdfile, "rb"))
    t = gsd.hoomd.HOOMDTrajectory(f)
    snap = t[0]
    if rmax is None:
        rmax = max(snap.configuration.box[:3]) / 2 - 1

    rdf = freud.density.RDF(bins, rmax)

    if stop is None:
        stop = len(t) - 1
    if start < 0:
        start += len(t) - 1
    for frame in range(start, stop):
        snap = t[frame]
        box = freud.box.Box(*snap.configuration.box)
        A_pos = snap.particles.position[
                snap.particles.typeid == snap.particles.types.index(A_name)
                ]
        pos = A_pos
        if A_name != B_name:
            B_pos = snap.particles.position[
                    snap.particles.typeid == snap.particles.types.index(B_name)
                    ]
            pos = np.concatenate((A_pos, B_pos))

        n_query = freud.locality.AABBQuery.from_system((
            box, pos
        ))
        rdf.compute(n_query, reset=False)
    return rdf


def get_compound_rdf(compound, A_name, B_name, rmax=None, bins=50, rdf=None):
    """
    This function calculates the radial distribution function given
    an mbuild compound, the names of the particles, and the dimensions of the box.

    Parameters
    ----------
    compound : CG_Compound
    A_name, B_name : str, name(s) of particle.name in compound
    rmax : float, maximum radius to consider. (default None)
        If none is given it'll be the minimum box length / 4
    rdf : freud.density.RDF, if provided, this function will accumulate an average rdf,
        otherwise it will provide the rdf only for the given compound. (default None)

    Returns
    -------
    freud.density.RDF
    """

    A_pos = compound.xyz[compound.get_name_inds(A_name), :]
    pos = A_pos
    if A_name != B_name:
        B_pos = compound.xyz[compound.get_name_inds(B_name), :]
        pos = np.concatenate((A_pos, B_pos))
    try:
        compound.box.lengths[0]
    except AttributeError(
        "No box found. Make sure you are using " + "CG_Compound and not mbuild.Compound"
    ):
        return
    except TypeError("Box has not been set"):
        return

    if rmax is None:
        rmax = min(compound.box.lengths) / 4
    if rdf is None:
        rdf = freud.density.RDF(bins, rmax)

    box = mb_to_freud_box(compound.box)
    n_query = freud.locality.AABBQuery.from_system((box, pos))

    rdf.compute(n_query, reset=False)
    return rdf


def map_good_on_bad(good_mol, bad_mol):
    """
    This function takes a correctly-typed (good) and a poorly-typed (bad)
    pybel molecule and transfers the bond and atom typing from the good to
    the bad molecule but retains the atom positions.
    It assumes that both molecules have the same number of particles and
    they maintain their order.
    Changes:
    atom- type, isaromatic
    bond- order, isaromatic

    Parameters
    ----------
    good_mol, bad_mol : pybel.Molecule

    Returns
    -------
    pybel.Molecule
    """

    for i in range(1, good_mol.OBMol.NumAtoms()):
        good_atom = good_mol.OBMol.GetAtom(i)
        bad_atom = bad_mol.OBMol.GetAtom(i)
        bad_atom.SetType(good_atom.GetType())
        bad_atom.SetAromatic(good_atom.IsAromatic())

    for i in range(1, good_mol.OBMol.NumBonds()):
        good_bond = good_mol.OBMol.GetBond(i)
        bad_bond = bad_mol.OBMol.GetBond(i)
        bad_bond.SetBondOrder(good_bond.GetBondOrder())
        bad_bond.SetAromatic(good_bond.IsAromatic())

    return bad_mol


def save_mol_to_file(good_mol, filename):
    """
    This function takes a correctly-typed (good) pybel molecule and saves
    the bond and atom typing to a file for later use.

    Parameters
    ----------
    good_mol : pybel.Molecule
    filename : str, name of file

    use map_file_on_bad() to use this file
    """

    with open(filename, "w") as f:
        for i in range(1, good_mol.OBMol.NumAtoms()):
            good_atom = good_mol.OBMol.GetAtom(i)
            f.write(f"{good_atom.GetType()}   {good_atom.IsAromatic()}\n")

        for i in range(1, good_mol.OBMol.NumBonds()):
            good_bond = good_mol.OBMol.GetBond(i)
            f.write(f"{good_bond.GetBO()}   {good_bond.IsAromatic()}\n")


def map_file_on_bad(filename, bad_mol):
    """
    This function takes a filename containing correctly-typed and a poorly-typed (bad)
    pybel molecule and transfers the bond and atom typing from the good to
    the bad molecule but retains the atom positions.
    It assumes that both molecules have the same number of particles and
    they maintain their order.
    Changes:
    atom- type, isaromatic
    bond- order, isaromatic

    Parameters
    ----------
    filename : str, generated using save_mol_to_file()
    bad_mol : pybel.Molecule

    Returns
    -------
    pybel.Molecule
    """
    with open(filename, "r") as f:
        lines = f.readlines()
    atoms = []
    bonds = []
    for line in lines:
        one, two = line.split()
        if not one.isdigit():
            atoms.append((one, two))
        else:
            bonds.append((one, two))

    for i in range(1, bad_mol.OBMol.NumAtoms()):
        bad_atom = bad_mol.OBMol.GetAtom(i)
        bad_atom.SetType(atoms[i - 1][0])
        if atoms[i - 1][1] == "True":
            bad_atom.SetAromatic()
        else:
            bad_atom.UnsetAromatic()

    for i in range(1, bad_mol.OBMol.NumBonds()):
        bad_bond = bad_mol.OBMol.GetBond(i)
        bad_bond.SetBO(int(bonds[i - 1][0]))
        if bonds[i - 1][1] == "True":
            bad_bond.SetAromatic()
        else:
            bad_bond.UnsetAromatic()

    return bad_mol


def has_number(string):
    """
    Returns True if string contains a number.
    Else returns False.
    """
    return bool(re.search("[0-9]", string))


def has_common_member(set_a, tup):
    """
    return True if set_a (set) and tup (tuple) share a common member
    else return False
    """
    set_b = set(tup)
    return set_a & set_b


def cg_comp(comp, bead_inds):
    """
    given an mbuild compound and bead_inds(list of tup)
    return coarse-grained mbuild compound
    """
    cg_compound = CG_Compound()
    cg_compound.box = comp.box

    for bead, smarts, bead_name in bead_inds:
        bead_xyz = comp.xyz[bead, :]
        avg_xyz = np.mean(bead_xyz, axis=0)
        bead = mb.Particle(name=bead_name, pos=avg_xyz)
        bead.smarts_string = smarts
        cg_compound.add(bead)
    return cg_compound


def cg_bonds(comp, cg_compound, beads):
    """
    add bonds based on bonding in aa compound
    return bonded mbuild compound
    """
    bonds = comp.get_bonds()
    bead_bonds = []
    for i, (bead_i, _, _) in enumerate(beads[:-1]):
        for j, (bead_j, _, _) in enumerate(beads[(i + 1) :]):
            for pair in bonds:
                if (pair[0] in bead_i) and (pair[1] in bead_j):
                    bead_bonds.append((i, j + i + 1))
                if (pair[1] in bead_i) and (pair[0] in bead_j):
                    bead_bonds.append((i, j + i + 1))
    for pair in bead_bonds:
        bond_pair = [
            particle
            for i, particle in enumerate(cg_compound.particles())
            if i == pair[0] or i == pair[1]
        ]
        cg_compound.add_bond(bond_pair)
    return cg_compound


def num2str(num):
    """
    Returns a capital letter for positive integers up to 701
    e.g. num2str(0) = 'A'
    """
    if num < 26:
        return chr(num + 65)
    return "".join([chr(num // 26 + 64), chr(num % 26 + 65)])


def coarse(mol, bead_list):
    """
    Creates a coarse-grained (CG) compound given a starting structure and
    smart strings for desired beads.

    Parameters
    ----------
    mol : pybel.Molecule
    bead_list : list of tuples of strings, desired bead name
    followed by SMARTS string of that bead

    Returns
    -------
    CG_Compound
    """
    matches = []
    for i, item in enumerate(bead_list):
        bead_name, smart_str = item
        smarts = pybel.Smarts(smart_str)
        if not smarts.findall(mol):
            print(f"{smart_str} not found in compound!")
        for group in smarts.findall(mol):
            group = tuple(i - 1 for i in group)
            matches.append((group, smart_str, bead_name))

    seen = set()
    bead_inds = []
    for group, smarts, name in matches:
        # smart strings for rings can share atoms
        # add bead regardless of whether it was seen
        if has_number(smarts):
            for atom in group:
                seen.add(atom)
            bead_inds.append((group, smarts, name))
        # alkyl chains should be exclusive
        else:
            if has_common_member(seen, group):
                pass
            else:
                for atom in group:
                    seen.add(atom)
                bead_inds.append((group, smarts, name))

    n_atoms = mol.OBMol.NumHvyAtoms()
    if n_atoms != len(seen):
        print(
            "WARNING: Some atoms have been left out of coarse-graining!"
        )  # TODO make this more informative

    comp = CG_Compound.from_pybel(mol)
    cg_compound = cg_comp(comp, bead_inds)
    cg_compound = cg_bonds(comp, cg_compound, bead_inds)

    cg_compound.atomistic = comp

    return cg_compound


amber_dict = {
    "c": "C",
    "c1": "C",
    "c2": "C",
    "c3": "C",
    "ca": "C",
    "cp": "C",
    "cq": "C",
    "cc": "C",
    "cd": "C",
    "ce": "C",
    "cf": "C",
    "cg": "C",
    "ch": "C",
    "cx": "C",
    "cy": "C",
    "cu": "C",
    "cv": "C",
    "h1": "H",
    "h2": "H",
    "h3": "H",
    "h4": "H",
    "h5": "H",
    "ha": "H",
    "hc": "H",
    "hn": "H",
    "ho": "H",
    "hp": "H",
    "hs": "H",
    "hw": "H",
    "hx": "H",
    "f": "F",
    "cl": "Cl",
    "br": "Br",
    "i": "I",
    "n": "N",
    "n1": "N",
    "n2": "N",
    "n3": "N",
    "n4": "N",
    "na": "N",
    "nb": "N",
    "nc": "N",
    "nd": "N",
    "ne": "N",
    "nf": "N",
    "nh": "N",
    "no": "N",
    "o": "O",
    "oh": "O",
    "os": "O",
    "ow": "O",
    "p2": "P",
    "p3": "P",
    "p4": "P",
    "p5": "P",
    "pb": "P",
    "pc": "P",
    "pd": "P",
    "pe": "P",
    "pf": "P",
    "px": "P",
    "py": "P",
    "s": "S",
    "s2": "S",
    "s4": "S",
    "s6": "S",
    "sh": "S",
    "ss": "S",
    "sx": "S",
    "sy": "S",
}


# features SMARTS
features_dict = {
    "thiophene": "c1sccc1",
    "thiophene_F": "c1scc(F)c1",
    "alkyl_3": "CCC",
    "benzene": "c1ccccc1",
    "splitring1": "csc",
    "splitring2": "cc",
    "twobenzene": "c2ccc1ccccc1c2",
    "ring_F": "c1sc2c(scc2c1F)",
    "ring_3": "c3sc4cc5ccsc5cc4c3",
    "chain1": "OCC(CC)CCCC",
    "chain2": "CCCCC(CC)COC(=O)",
    "cyclopentadiene": "C1cccc1",
    "c4": "cC(c)(c)c",
    "cyclopentadienone": "C=C1C(=C)ccC1=O",
}

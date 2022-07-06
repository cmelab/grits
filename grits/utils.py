"""Utility functions for GRiTS."""
import json
import re

import ele
import freud
import numpy as np
from ele import element_from_symbol
from mbuild.box import Box
from mbuild.compound import Compound, Particle


class NumpyEncoder(json.JSONEncoder):
    """Serializer for numpy objects."""

    def default(self, obj):
        """Overwrite the default for numpy arrays and data types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


def comp_from_snapshot(snapshot, indices, scale=1.0, shift_coords=True):
    """Convert particles by indices from a Snapshot to a Compound.

    Parameters
    ----------
    snapshot : gsd.hoomd.Snapshot
        Snapshot from which to build the mbuild Compound.
    indices : np.ndarray
        Indices of the particles to be added to the compound.
    scale : float, default 1.0
        Value by which to scale the length values
    shift_coords : bool, default True
        Whether to shift the coords by half the box lengths

    Returns
    -------
    comp : mbuild.Compound

    Note
    ----
    GSD snapshots center their boxes on the origin (0,0,0), so the compound is
    shifted by half the box lengths
    """
    comp = Compound()
    bond_array = snapshot.bonds.group
    n_atoms = snapshot.particles.N

    # gsd / hoomd v3
    box = np.asarray(snapshot.configuration.box)
    comp.box = Box.from_lengths_tilt_factors(
        lengths=box[:3] * scale, tilt_factors=box[3:]
    )

    if shift_coords:
        # GSD and HOOMD snapshots center their boxes on the origin (0,0,0)
        shift = np.array(comp.box.lengths) / 2
    else:
        shift = np.zeros(3)
    particle_dict = {}
    # Add particles
    for i in range(n_atoms):
        if i in indices:
            name = snapshot.particles.types[snapshot.particles.typeid[i]]
            xyz = snapshot.particles.position[i] * scale + shift

            atom = Particle(name=name, pos=xyz)
            comp.add(atom, label=str(i))
            particle_dict[i] = atom

    # Add bonds
    for i, j in snapshot.bonds.group:
        if i in indices and j in indices:
            comp.add_bond([particle_dict[i], particle_dict[j]])
    return comp


def snap_molecules(snap):
    """Get the molecule indices based on bonding in a gsd.hoomd.Snapshot."""
    system = freud.AABBQuery.from_system(snap)
    n_query_points = n_points = snap.particles.N
    query_point_indices = snap.bonds.group[:, 0]
    point_indices = snap.bonds.group[:, 1]
    distances = system.box.compute_distances(
        system.points[query_point_indices], system.points[point_indices]
    )
    nlist = freud.NeighborList.from_arrays(
        n_query_points, n_points, query_point_indices, point_indices, distances
    )
    cluster = freud.cluster.Cluster()
    cluster.compute(system=system, neighbors=nlist)
    return cluster.cluster_idx


def align(compound, particle, towards_compound, around=None):
    """Spin a compound such that particle points at towards_compound.

    Parameters
    ----------
    compound : mbuild.Compound or CG_Compound,
        The compound to align
    particle : mbuild.Compound or CG_Compound,
        The particle to point at towards_compound. Child of compound.
    towards_compound : mbuild.Compound or CG_Compound,
        The compound to point towards.
    around : numpy.ndarray, default None
        The unit vector around which the compound is spun. If None is given,
        an orthogonal vector is chosen.

    Returns
    -------
    numpy.ndarray
        The unit vector about which the compound is rotated
    """
    # Find the unit vector from compound center to particle
    vec = np.array(particle.pos - compound.center)
    sep = np.linalg.norm(vec)
    comp_to_part = vec / sep

    # Get the unit vector between the two centers
    # end - start: from 1 -> 2
    vec = np.array(towards_compound.center - compound.center)
    sep = np.linalg.norm(vec)
    towards_to_comp = vec / sep

    if around is None:
        # Next get a vector orthogonal to both vectors,
        # this is the vector around which the compound is spun
        around = np.cross(comp_to_part, towards_to_comp)
    # and the angle between the two vectors (in rad)
    angle = np.arccos(np.dot(comp_to_part, towards_to_comp))

    compound.spin(angle, around)
    return around


def get_bonds(compound):
    """Convert Particle instances in bond_graph.edges to their indices.

    See :py:meth:`CG_Compound.bond_graph.edges`.

    Parameters
    ----------
    compound : mbuild.Compound or CG_Compound
       Compound from which to get the indexed bond graph.

    Returns
    -------
    list[tuple(int, int)]
        Sorted list of bonded particle indices
    """
    particle_list = [p for p in compound]
    bonds = []
    for tup in compound.bond_graph.edges():
        bonds.append(tuple(sorted(map(particle_list.index, tup))))
    # This sorting is required for coarse-graining
    bonds.sort(key=lambda tup: (tup[0], tup[1]))
    return bonds


def get_bonded(compound, particle):
    """Get particles bonded to the given particle in a compound.

    Parameters
    ----------
    compound : mbuild.Compound or CG_Compound
        Compound containing particle
    particle : mbuild.Particle or Bead
        Particle to which the directly bonded neighbors are requested

    Returns
    -------
    list[mbuild.Particle]
        The bonded particles.
    """

    def is_particle(i, j):
        if i is particle:
            return j
        elif j is particle:
            return i
        else:
            return False

    xs = [is_particle(i, j) for i, j in compound.bonds() if is_particle(i, j)]
    # The following logic won't be necessary when bond graph is deterministic
    # https://github.com/mosdef-hub/mbuild/issues/895
    # and instead we can just do:
    return xs
    # return [x for _,x in sorted(zip([get_index(compound,i) for i in xs],xs))]


def get_index(compound, particle):
    """Get the index of a Particle in the Compound.

    The particle can be accessed like compound[index].

    Parameters
    ----------
    compound : mbuild.Compound or CG_Compound
        Compound which contains particle
    particle : mbuild.Particle or Bead
        Particle for which to fetch the index

    Returns
    -------
    int
        The particle index
    """
    return [p for p in compound].index(particle)


def get_hydrogen(compound, particle):
    """Get the first hydrogen attached to particle.

    Parameters
    ----------
    compound : mbuild.Compound or CG_Compound
        Compound which contains particle
    particle : mbuild.Particle or Bead
        Particle from which to remove a hydrogen
    """
    hydrogens = [i for i in get_bonded(compound, particle) if i.name == "H"]
    if hydrogens:
        for p in compound:
            if p in hydrogens:
                return p
    else:
        return None


def has_number(string):
    """Determine whether a string contains a number.

    Parameters
    ----------
    string : str
        String which may contain a number

    Returns
    -------
    bool
        Whether the string contains a number.
    """
    return bool(re.search("[0-9]", string))


def has_common_member(it_a, it_b):
    """Determine if two iterables share a common member.

    Parameters
    ----------
    it_a, it_b : iterable object
        Iterable objects to compare

    Returns
    -------
    bool
        Whether the object share a common member.
    """
    return set(it_a) & set(it_b)


def num2str(num):
    """Convert a number to a string.

    Convert positive integers up to 701 into a capital letter (or letters).

    Parameters
    ----------
    num : int
        Number to convert

    Examples
    --------
    >>> num2str(0)
    'A'
    >>> num2str(25)
    'Z'
    >>> num2str(25)
    'AA'

    Returns
    -------
    str
        The string conversion of the number
    """
    if num < 26:
        return chr(num + 65)
    return "".join([chr(num // 26 + 64), chr(num % 26 + 65)])


amber_dict = {
    "c": ele.element_from_symbol("C"),
    "c1": ele.element_from_symbol("C"),
    "c2": ele.element_from_symbol("C"),
    "c3": ele.element_from_symbol("C"),
    "ca": ele.element_from_symbol("C"),
    "cp": ele.element_from_symbol("C"),
    "cq": ele.element_from_symbol("C"),
    "cc": ele.element_from_symbol("C"),
    "cd": ele.element_from_symbol("C"),
    "ce": ele.element_from_symbol("C"),
    "cf": ele.element_from_symbol("C"),
    "cg": ele.element_from_symbol("C"),
    "ch": ele.element_from_symbol("C"),
    "cx": ele.element_from_symbol("C"),
    "cy": ele.element_from_symbol("C"),
    "cu": ele.element_from_symbol("C"),
    "cv": ele.element_from_symbol("C"),
    "h1": ele.element_from_symbol("H"),
    "h2": ele.element_from_symbol("H"),
    "h3": ele.element_from_symbol("H"),
    "h4": ele.element_from_symbol("H"),
    "h5": ele.element_from_symbol("H"),
    "ha": ele.element_from_symbol("H"),
    "hc": ele.element_from_symbol("H"),
    "hn": ele.element_from_symbol("H"),
    "ho": ele.element_from_symbol("H"),
    "hp": ele.element_from_symbol("H"),
    "hs": ele.element_from_symbol("H"),
    "hw": ele.element_from_symbol("H"),
    "hx": ele.element_from_symbol("H"),
    "f": ele.element_from_symbol("F"),
    "cl": ele.element_from_symbol("Cl"),
    "br": ele.element_from_symbol("Br"),
    "i": ele.element_from_symbol("I"),
    "n": ele.element_from_symbol("N"),
    "n1": ele.element_from_symbol("N"),
    "n2": ele.element_from_symbol("N"),
    "n3": ele.element_from_symbol("N"),
    "n4": ele.element_from_symbol("N"),
    "na": ele.element_from_symbol("N"),
    "nb": ele.element_from_symbol("N"),
    "nc": ele.element_from_symbol("N"),
    "nd": ele.element_from_symbol("N"),
    "ne": ele.element_from_symbol("N"),
    "nf": ele.element_from_symbol("N"),
    "nh": ele.element_from_symbol("N"),
    "no": ele.element_from_symbol("N"),
    "o": ele.element_from_symbol("O"),
    "oh": ele.element_from_symbol("O"),
    "os": ele.element_from_symbol("O"),
    "ow": ele.element_from_symbol("O"),
    "p2": ele.element_from_symbol("P"),
    "p3": ele.element_from_symbol("P"),
    "p4": ele.element_from_symbol("P"),
    "p5": ele.element_from_symbol("P"),
    "pb": ele.element_from_symbol("P"),
    "pc": ele.element_from_symbol("P"),
    "pd": ele.element_from_symbol("P"),
    "pe": ele.element_from_symbol("P"),
    "pf": ele.element_from_symbol("P"),
    "px": ele.element_from_symbol("P"),
    "py": ele.element_from_symbol("P"),
    "s": ele.element_from_symbol("S"),
    "s2": ele.element_from_symbol("S"),
    "s4": ele.element_from_symbol("S"),
    "s6": ele.element_from_symbol("S"),
    "sh": ele.element_from_symbol("S"),
    "ss": ele.element_from_symbol("S"),
    "sx": ele.element_from_symbol("S"),
    "sy": ele.element_from_symbol("S"),
}

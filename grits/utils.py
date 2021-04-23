"""Utility functions for GRiTS."""
import re

import numpy as np


def get_bonds(compound):
    """Convert Particle instances in bond_graph.bond_edges to their indices.

    Parameters
    ----------
    compound : mbuild.Compound or CG_Compound
       Compound from which to get the indexed bond graph

    Returns
    -------
    sorted list of tuples of bonded particle indices
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
    list of mbuild.Particles
    """

    def is_particle(i, j):
        if i is particle:
            return j
        elif j is particle:
            return i
        else:
            return False

    return [is_particle(i, j) for i, j in compound.bonds() if is_particle(i, j)]


def get_index(compound, particle):
    """Get the index of a Particle in the Compound.

    The particle can be accessed like compound[index].

    Parameters
    ----------
    compound: mbuild.Compound or CG_Compound
        Compound which contains particle
    particle: mbuild.Particle or Bead
        Particle for which to fetch the index

    Returns
    -------
    int
    """
    return [p for p in compound].index(particle)


def remove_hydrogen(compound, particle):
    """Remove one hydrogen attached to particle.

    Hydrogen particle name must be "H". If no hydrogen is bonded to particle,
    this function will do nothing.

    Parameters
    ----------
    compound: mbuild.Compound or CG_Compound
        Compound which contains particle
    particle: mbuild.Particle or Bead
        Particle from which to remove a hydrogen
    """
    hydrogens = [i for i in get_bonded(compound, particle) if i.name == "H"]
    if hydrogens:
        compound.remove(hydrogens[0])


def has_number(string):
    """Determine whether a string contains a number.

    Parameters
    ----------
    string: str
        string which may contain a number

    Returns
    -------
    bool
    """
    return bool(re.search("[0-9]", string))


def has_common_member(it_a, it_b):
    """Determine if two iterables share a common member.

    Parameters
    ----------
    it_a, it_b: iterable objects
        iterable objects to compare

    Returns
    -------
    bool
    """
    return set(it_a) & set(it_b)


def num2str(num):
    """Convert a number to a string.

    Convert positive integers up to 701 into a capital letter (or letters).

    Parameters
    ----------
    num: int
        number to convert

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
    """
    if num < 26:
        return chr(num + 65)
    return "".join([chr(num // 26 + 64), chr(num % 26 + 65)])


def distance(pos1, pos2):
    """Calculate euclidean distance between two points.

    Parameters
    ----------
    pos1, pos2 : numpy.ndarray,
        x, y, and z coordinates (2D also works)

    Returns
    -------
    float
    """
    return np.linalg.norm(pos1 - pos2)


def v_distance(pos1, pos2):
    """Calculate euclidean distances between all points in pos1 and pos2.

    Parameters
    ----------
    pos1 : numpy.ndarray, shape=(N,3)
        array of x, y, and z coordinates
    pos2 : numpy.ndarray, shape=(3,)
        x, y, and z coordinates

    Notes
    -----
    `pos1` and `pos2` are interchangeable, but to correctly calculate the
    distances only one of them can be a 2D array.

    Returns
    -------
    (N,) numpy.ndarray
    """
    return np.linalg.norm(pos1 - pos2, axis=1)

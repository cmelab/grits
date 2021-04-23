import re

import numpy as np


def get_bonds(compound):
    """
    Translates bond_graph.bond_edges to particle indices in compound

    Returns
    -------
    list of tuples of bonded atom indices sorted
    """
    particle_list = [p for p in compound]
    bonds = []
    for tup in compound.bond_graph.edges():
        bonds.append(tuple(sorted(map(particle_list.index, tup))))
    # This sorting is required for coarse-graining
    bonds.sort(key=lambda tup: (tup[0], tup[1]))
    return bonds


def get_bonded(compound, particle):
    """
    Returns a list of particles bonded to the given particle
    in a compound

    Parameters
    ----------
    compound : mbuild.Compound, compound containing particle
    particle : mbuild.Particle, particle to which we want to
               find the bonded neighbors.

    Returns
    -------
    list of mbuild.Particles
    """
    def is_particle(i,j):
        if i is particle:
            return j
        elif j is particle:
            return i
        else:
            return False
    return [is_particle(i,j) for i,j in compound.bonds() if is_particle(i,j)]


def get_index(compound, particle):
    """
    Get the index of a particle in the compound so that the particle
    can be accessed like compound[index]

    Parameters
    ----------
    compound: mbuild.Compound, compound which contains particle
    particle: mbuild.Particle, particle for which to fetch the index

    Returns
    -------
    int
    """
    return [p for p in compound].index(particle)


def remove_hydrogen(compound, particle):
    """
    Remove a hydrogen attached to particle. Particle name must be "H".
    If no hydrogen is bonded to particle, do nothing.

    Parameters
    ----------
    compound: mbuild.Compound, compound which contains particle
    particle: mbuild.Particle, particle from which to remove a hydrogen
    """
    hydrogens = [i for i in get_bonded(compound, particle) if i.name == "H"]
    if hydrogens:
        compound.remove(hydrogens[0])


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


def num2str(num):
    """
    Returns a capital letter for positive integers up to 701
    e.g. num2str(0) = 'A'
    """
    if num < 26:
        return chr(num + 65)
    return "".join([chr(num // 26 + 64), chr(num % 26 + 65)])


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

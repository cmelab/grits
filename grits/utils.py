"""Utility functions for GRiTS."""
import re

import numpy as np


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

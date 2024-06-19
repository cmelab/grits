"""Utility functions for GRiTS."""

import json
import re
import warnings

import ele
import freud
import numpy as np
import rowan
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


def comp_from_snapshot(snapshot, indices, length_scale=1.0, mass_scale=1.0):
    """Convert particles by indices from a Snapshot to a Compound.

    Parameters
    ----------
    snapshot : gsd.hoomd.Frame
        Snapshot from which to build the mbuild Compound.
    indices : np.ndarray
        Indices of the particles to be added to the compound.
    length_scale : float, default 1.0
        Value by which to scale the length values
    mass_scale : float, default 1.0
        Value by which to scale the mass values

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
        lengths=box[:3] * length_scale, tilt_factors=box[3:]
    )

    # GSD and HOOMD snapshots center their boxes on the origin (0,0,0)
    shift = np.array(comp.box.lengths) / 2
    particle_dict = {}
    # Add particles
    for i in range(n_atoms):
        if i in indices:
            name = snapshot.particles.types[snapshot.particles.typeid[i]]
            xyz = snapshot.particles.position[i] * length_scale + shift
            mass = snapshot.particles.mass[i] * mass_scale
            element = ele.element_from_symbol(name)
            atom = Particle(name=name, pos=xyz, mass=mass, element=element)
            comp.add(atom, label=str(i))
            particle_dict[i] = atom

    # Add bonds
    for i, j in snapshot.bonds.group:
        if i in indices and j in indices:
            comp.add_bond([particle_dict[i], particle_dict[j]])
    return comp


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


def get_heavy_atoms(particles):
    """Filter out hydrogens for axis-angle calculation.

    Returns arrays of only heavy atoms' positions and masses,
    given a gsd.frame.particles object. Used in Aniso mapping.

    Parameters
    ----------
    particles : gsd.frame.particles
        Particles from all atom gsd frame.

    Returns
    -------
    heavy_partpos : numpy array
        Array of all positions of heavy atoms
    heavy_partmass : numpy array
        Array of all masses of heavy atoms
    """
    partpos = particles.position
    partmass = particles.mass
    partelem = particles.typeid
    heavy_atom_indicies = np.where(partelem != 1)[0]
    heavy_partpos = partpos[heavy_atom_indicies]
    heavy_partmass = partmass[heavy_atom_indicies]
    return heavy_partpos, heavy_partmass


def get_major_axis(positions_arr):
    """Find the major axis for GB CG representation.

    Used in axis-angle orientation representation.

    Parameters
    ----------
        positions_arr : numpy array
            N_particlesx3 numpy array of particle positions
            to map to one aniso bead.
        elements_arr : list
            List of length N_particles containing particle elements

    Returns
    -------
        major_axis : numpy array
            array designating vector of major axis of Gay-Berne particle
        particle_indicies : tuple of ints
            tuple of two particle indices used to calculate major axis vector
    """
    major_axis = None
    max_dist = 0
    AB_indicies = (None, None)
    for i, x0 in enumerate(positions_arr):
        for j, x1 in enumerate(positions_arr[i + 1 :]):
            vect = x1 - x0
            dist = np.linalg.norm(vect)
            if dist > max_dist:
                max_dist = dist
                major_axis = vect
                # adjust j for loop stride
                AB_indicies = (i, j + i + 1)
    return major_axis, AB_indicies


def get_com(particle_positions, particle_masses):
    """Calculate center of mass coordinates.

       Given a set of particle positions and masses, find
       their center of mass.
       Arrays must be of same dimension.

    Parameters
    ----------
        particle_positions : numpy array
            N_particlesx3 numpy array of particle positions (x,y,z)
        particle_masses : numpy array
            N_particlesx0 numpy array of particle masses

    Returns
    -------
        center_of_mass : numpy array
            3x0 numpy array of center of mass coordinates
    """
    M = np.sum(particle_masses)
    weighted_positions = particle_positions * particle_masses[:, np.newaxis]
    center_of_mass = np.sum(weighted_positions / M, axis=0)
    return center_of_mass


def get_quaternion(n1, n0=np.array([0, 0, 1])):
    """Calculate rotation quaternion from axis vectors.

    Calculate axis and angle of rotation given
    two planes' normal vectors, which is then used
    to calculate the quaternions for HOOMD.

    Parameters
    ----------
        n1 : numpy array
            numpy array that is the major axis vector.
        n0 : numpy array
            numpy array that is used to define the default quaternion.
            Defaults to the Z-axis.

    Returns
    -------
        quaternion : numpy array
            numpy array that tells the position of the monomer in units
            of a quaternion.
    """
    if n1 is None:  # one atom in this bead -> default quaternion
        warnings.warn(
            "get_quaternion was called with None as input!\n\
                      Returning default orientation."
        )
        return np.array([0, 0, 0, 1])
    V_axis = np.cross(n0, n1)
    theta_numerator = np.dot(n0, n1)
    theta_denominator = np.linalg.norm(n0) * np.linalg.norm(n1)
    theta_rotation = np.arccos(theta_numerator / theta_denominator)
    quaternion = rowan.from_axis_angle(V_axis, theta_rotation)
    return quaternion


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

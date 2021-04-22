import os
import tempfile
from collections import OrderedDict, defaultdict
from copy import deepcopy

import freud
import gsd
import gsd.hoomd
import mbuild as mb
import numpy as np
from openbabel import pybel
from mbuild.exceptions import MBuildError
from mbuild.utils.io import import_, run_from_ipython
from oset import oset as OrderedSet
from parmed.periodic_table import Element

from grits.utils import (
        has_number, has_common_member, distance, v_distance, mb_to_freud_box
        )


class CG_Compound(mb.Compound):
    def __init__(self):
        super().__init__()
        self.box = None
        self.atomistic = None


    def convert_types(self, conversion_dict):
        """Convert type to element name.
        """
        for particle in self.particles():
            particle.name = conversion_dict[particle.name]

    def remove_hydrogens(self):
        """
        Remove all particles with name = "H" in the compound
        """
        self.remove([i for i in self.particles() if i.name == "H"])

    def get_molecules(self):
        """
        Translates bond_graph.connected_components to particle indices in compound

        Returns
        -------
        list of sets of connected atom indices
        """
        particle_list = [part for part in self.particles()]
        molecules = []
        for group in self.bond_graph.connected_components():
            molecules.append(set(map(particle_list.index, group)))
        return molecules

    def get_bonds(self):
        """
        Translates bond_graph.bond_edges to particle indices in compound

        Returns
        -------
        list of tuples of bonded atom indices sorted
        """
        particle_list = [part for part in self.particles()]
        bonds = []
        for tup in self.bond_graph.edges():
            bonds.append(tuple(sorted(map(particle_list.index, tup))))
        # This sorting is required for coarse-graining
        bonds.sort(key=lambda tup: (tup[0], tup[1]))
        return bonds


    def get_name_inds(self, name):
        """
        Find indices of particles in compound where particle.name matches given name

        Parameters
        ----------
        name : str, particle.name in mb.Compound

        Returns
        -------
        list of particles indices which match name
        """
        return [i for i, part in enumerate(self.particles()) if part.name == name]


    def tuple_to_names(self, tup):
        """
        Get the names of particle indices passed in as a tuple.

        Parameters
        ----------
        tup : tuple of ints, particle indices

        Returns
        -------
        tuple of strings, particle.name of given indices
        """
        particles = [part for part in self.particles()]

        types = []
        for index in tup:
            types.append(particles[index].name)
        return tuple(sorted(types))


    @classmethod
    def from_mbuild(cls, compound):
        """
        Instantiates a CG_Compound and follows mb.Compound.deep_copy
        to copy particles and bonds to CG_Compound

        Parameters
        ----------
        compound : mb.Compound to be compied

        Returns
        -------
        CG_Compound
        """

        comp = cls()

        clone_dict = {}
        comp.name = deepcopy(compound.name)
        comp.periodicity = deepcopy(compound.periodicity)
        comp._pos = deepcopy(compound._pos)
        comp.port_particle = deepcopy(compound.port_particle)
        comp._check_if_contains_rigid_bodies = deepcopy(
            compound._check_if_contains_rigid_bodies
        )
        comp._contains_rigid = deepcopy(compound._contains_rigid)
        comp._rigid_id = deepcopy(compound._rigid_id)
        comp._charge = deepcopy(compound._charge)

        if compound.children is None:
            comp.children = None
        else:
            comp.children = OrderedSet()
        # Parent should be None initially.
        comp.parent = None
        comp.labels = OrderedDict()
        comp.referrers = set()
        comp.bond_graph = None
        for p in compound.particles():
            new_particle = mb.Particle(name=p.name, pos=p.xyz.flatten())
            comp.add(new_particle)
            clone_dict[p] = new_particle

        for c1, c2 in compound.bonds():
            try:
                comp.add_bond((clone_dict[c1], clone_dict[c2]))
            except KeyError:
                raise MBuildError(
                    "Cloning failed. Compound contains bonds to "
                    "Particles outside of its containment hierarchy."
                )
        return comp


    def visualize(self, show_ports=False, backend='py3dmol',
            color_scheme={}, show_atomistic=False, scale=1.0): # pragma: no cover
        """
        Visualize the Compound using py3dmol (default) or nglview.
        Allows for visualization of a Compound within a Jupyter Notebook.
        Parameters
        ----------
        show_ports : bool, optional, default=False
            Visualize Ports in addition to Particles
        backend : str, optional, default='py3dmol'
            Specify the backend package to visualize compounds
            Currently supported: py3dmol, nglview
        color_scheme : dict, optional
            Specify coloring for non-elemental particles
            keys are strings of the particle names
            values are strings of the colors
            i.e. {'_CGBEAD': 'blue'}
        NOTE!: Only py3dmol will work with CG_Compounds
        """
        viz_pkg = {'nglview': self._visualize_nglview,
                'py3dmol': self._visualize_py3dmol}
        if run_from_ipython():
            if backend.lower() in viz_pkg:
                return viz_pkg[backend.lower()](show_ports=show_ports,
                        color_scheme=color_scheme, show_atomistic=show_atomistic, scale=scale)
            else:
                raise RuntimeError("Unsupported visualization " +
                        "backend ({}). ".format(backend) +
                        "Currently supported backends include nglview and py3dmol")

        else:
            raise RuntimeError('Visualization is only supported in Jupyter '
                               'Notebooks.')


    def _visualize_py3dmol(self,
            show_ports=False,
            color_scheme={},
            show_atomistic=False,
            scale=1.0):
        """
        Visualize the Compound using py3Dmol.
        Allows for visualization of a Compound within a Jupyter Notebook.
        Modified to show atomistic elements (translucent) with larger CG beads.

        Parameters
        ----------
        show_ports : bool, optional, default=False
            Visualize Ports in addition to Particles
        color_scheme : dict, optional
            Specify coloring for non-elemental particles
            keys are strings of the particle names
            values are strings of the colors
            i.e. {'_CGBEAD': 'blue'}
        show_atomistic : show the atomistic structure stored in CG_Compound.atomistic

        Returns
        ------
        view : py3Dmol.view
        """
        py3Dmol = import_("py3Dmol")

        atom_names = []

        if self.atomistic is not None and show_atomistic:
            atomistic = mb.clone(self.atomistic)
            for particle in atomistic.particles():
                if not particle.name:
                    particle.name = "UNK"
                else:
                    if (particle.name != 'Compound') and (particle.name != 'CG_Compound'):
                        atom_names.append(particle.name)

        coarse = mb.clone(self)
        modified_color_scheme = {}
        for name, color in color_scheme.items():
            # Py3dmol does some element string conversions,
            # first character is as-is, rest of the characters are lowercase
            new_name = name[0] + name[1:].lower()
            modified_color_scheme[new_name] = color
            modified_color_scheme[name] = color

        cg_names = []
        for particle in coarse.particles():
            if not particle.name:
                particle.name = "UNK"
            else:
                if (particle.name != 'Compound') and (particle.name != 'CG_Compound'):
                    cg_names.append(particle.name)


        tmp_dir = tempfile.mkdtemp()

        view = py3Dmol.view()

        if atom_names:
            atomistic.save(
                os.path.join(tmp_dir, "atomistic_tmp.mol2"),
                show_ports=show_ports,
                overwrite=True,
            )

            # atomistic
            with open(os.path.join(tmp_dir, "atomistic_tmp.mol2"), "r") as f:
                view.addModel(f.read(), "mol2", keepH=True)

            if cg_names:
                opacity = 0.6
            else:
                opacity = 1.0

            view.setStyle(
                {
                    "stick": {"radius": 0.2 * scale, "opacity": opacity, "color": "grey"},
                    "sphere": {
                        "scale": 0.3 * scale,
                        "opacity": opacity,
                        "colorscheme": modified_color_scheme,
                    },
                }
            )

        # coarse
        if cg_names:
            coarse.save(
                os.path.join(tmp_dir, "coarse_tmp.mol2"),
                show_ports=show_ports,
                overwrite=True,
            )
            with open(os.path.join(tmp_dir, "coarse_tmp.mol2"), "r") as f:
                view.addModel(f.read(), "mol2", keepH=True)

            if self.atomistic is None:
                scale = 0.3 * scale
            else:
                scale = 0.7 * scale

            view.setStyle(
                {"atom": cg_names},
                {
                    "stick": {"radius": 0.2 * scale, "opacity": 1, "color": "grey"},
                    "sphere": {
                        "scale": scale,
                        "opacity": 1,
                        "colorscheme": modified_color_scheme,
                    },
                },
            )

        view.zoomTo()

        return view


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

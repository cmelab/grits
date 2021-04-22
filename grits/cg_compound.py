import os
import tempfile
from warnings import warn

import numpy as np
from openbabel import pybel
from mbuild.utils.io import import_, run_from_ipython
from mbuild import Compound, Particle, clone

from grits.utils import has_number, has_common_member, get_bonds


class CG_Compound(Compound):
    """
    Creates a coarse-grained (CG) compound given a starting structure and
    smart strings for desired beads.

    Parameters
    ----------
    mol : pybel.Molecule
    bead_list : list of tuples of strings, desired bead name
    followed by SMARTS string of that bead

    Attributes
    ----------
    atomistic
    bead_inds
    """
    def __init__(self, compound, bead_list):
        super().__init__()
        self.atomistic = compound

        mol = compound.to_pybel()
        mol.OBMol.PerceiveBondOrders()

        self.set_bead_inds(bead_list, mol)
        self.cg_particles()
        self.cg_bonds()

    def set_bead_inds(self, bead_list, mol):
        matches = []
        for i, item in enumerate(bead_list):
            bead_name, smart_str = item
            smarts = pybel.Smarts(smart_str)
            if not smarts.findall(mol):
                warn(f"{smart_str} not found in compound!")
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
            warn("Some atoms have been left out of coarse-graining!")
            # TODO make this more informative
        self.bead_inds = bead_inds

    def cg_particles(self):
        """
        given an mbuild compound and bead_inds(list of tup)
        return coarse-grained mbuild compound
        """
        for bead, smarts, bead_name in self.bead_inds:
            bead_xyz = self.atomistic.xyz[bead, :]
            avg_xyz = np.mean(bead_xyz, axis=0)
            bead = Particle(name=bead_name, pos=avg_xyz)
            bead.smarts_string = smarts
            self.add(bead)

    def cg_bonds(self):
        """
        add bonds based on bonding in aa compound
        return bonded mbuild compound
        """
        bonds = get_bonds(self.atomistic)
        bead_bonds = []
        for i, (bead_i, _, _) in enumerate(self.bead_inds[:-1]):
            for j, (bead_j, _, _) in enumerate(self.bead_inds[(i + 1) :]):
                for pair in bonds:
                    if (pair[0] in bead_i) and (pair[1] in bead_j):
                        bead_bonds.append((i, j + i + 1))
                    if (pair[1] in bead_i) and (pair[0] in bead_j):
                        bead_bonds.append((i, j + i + 1))
        for pair in bead_bonds:
            bond_pair = [
                p for i, p in enumerate(self.particles()) if i in pair
            ]
            self.add_bond(bond_pair)

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
            atomistic = clone(self.atomistic)
            for particle in atomistic.particles():
                if not particle.name:
                    particle.name = "UNK"
                else:
                    if (particle.name != 'Compound') and (particle.name != 'CG_Compound'):
                        atom_names.append(particle.name)

        coarse = clone(self)
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

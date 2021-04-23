"""GRiTS: Coarse-graining tools."""
import os
import tempfile
from warnings import warn

import numpy as np
from mbuild import Compound, clone
from mbuild.utils.io import run_from_ipython
from openbabel import pybel

from grits.utils import get_bonds, has_common_member, has_number


class CG_Compound(Compound):
    """Coarse-grained Compound.

    Wrapper for mbuild.Compound. Coarse-grained mapping can be specified using
    SMARTS grammar.

    Parameters
    ----------
    compound : mbuild.Compound
        fine-grain structure to be coarse-grained
    beads : list of tuples of strings
        list of pairs containing desired bead name followed by SMARTS string
        specification of that bead. For example:

        >>> beads = [("_B", "c1sccc1"), ("_S", "CCC")]

        would map a `"_B"` bead to any thiophene moiety (`"c1sccc1"`) found in
        the compound and an `"_S"` bead to a propyl moiety (`"CCC"`).

    Attributes
    ----------
    atomistic: mbuild.Compound
        the atomistic structure
    bead_inds: list of (tuple of ints, string, string)
        Each list item corresponds to the particle indices in that bead, the
        smarts string used to find that bead, and the name of the bead.

    Methods
    -------
    visualize(
        show_ports=False, color_scheme={}, show_atomistic=False, scale=1.0
        )
        Visualize the CG_Compound in a Jupyter Notebook.
    """

    def __init__(self, compound, beads):
        super(CG_Compound, self).__init__()
        self.atomistic = compound

        mol = compound.to_pybel()
        mol.OBMol.PerceiveBondOrders()

        self._set_bead_inds(beads, mol)
        self._cg_particles()
        self._cg_bonds()

    def _set_bead_inds(self, beads, mol):
        matches = []
        for i, item in enumerate(beads):
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

    def _cg_particles(self):
        """Set the beads in the coarse-structure."""
        for bead, smarts, bead_name in self.bead_inds:
            bead_xyz = self.atomistic.xyz[bead, :]
            avg_xyz = np.mean(bead_xyz, axis=0)
            bead = Bead(name=bead_name, pos=avg_xyz, smarts=smarts)
            self.add(bead)

    def _cg_bonds(self):
        """Set the bonds in the coarse structure."""
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
            bond_pair = [p for i, p in enumerate(self) if i in pair]
            self.add_bond(bond_pair)

    def visualize(
        self, show_ports=False, color_scheme={}, show_atomistic=False, scale=1.0
    ):  # pragma: no cover
        """Visualize the Compound using py3dmol.

        Allows for visualization of a Compound within a Jupyter Notebook.
        Modified to from mbuild.Compound.visualize to show atomistic elements
        (translucent) with larger CG beads.

        Parameters
        ----------
        show_ports : bool, default False
            Visualize Ports in addition to Particles
        color_scheme : dict, default {}
            Specify coloring for non-elemental particles keys are strings of
            the particle names values are strings of the colors:

            >>> {'_CGBEAD': 'blue'}

        show_atomistic : bool, default False
            Show the atomistic structure stored in CG_Compound.atomistic
        scale : float, default 1.0
            Scaling factor to modify the size of objects in the view.

        Returns
        -------
        view : py3Dmol.view
        """
        if not run_from_ipython():
            raise RuntimeError(
                "Visualization is only supported in Jupyter Notebooks."
            )
        import py3Dmol

        atom_names = []

        if self.atomistic is not None and show_atomistic:
            atomistic = clone(self.atomistic)
            for particle in atomistic:
                if not particle.name:
                    particle.name = "UNK"
                else:
                    if particle.name not in ("Compound", "CG_Compound"):
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
        for particle in coarse:
            if not particle.name:
                particle.name = "UNK"
            else:
                if particle.name not in ("Compound", "CG_Compound"):
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
                    "stick": {
                        "radius": 0.2 * scale,
                        "opacity": opacity,
                        "color": "grey",
                    },
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
                    "stick": {
                        "radius": 0.2 * scale,
                        "opacity": 1,
                        "color": "grey",
                    },
                    "sphere": {
                        "scale": scale,
                        "opacity": 1,
                        "colorscheme": modified_color_scheme,
                    },
                },
            )

        view.zoomTo()

        return view


class Bead(Compound):
    """Coarse-grained Bead.

    Wrapper for mbuild.Compound.

    Parameters
    ----------
    smarts : str, default None
        SMARTS string used to specify this Bead.

    Attributes
    ----------
    smarts : str
        SMARTS string used to specify this Bead.
    """

    def __init__(self, smarts=None, **kwargs):
        self.smarts = smarts
        super(Bead, self).__init__(**kwargs)

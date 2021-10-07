"""GRiTS: Coarse-graining tools."""
import json
import os
import tempfile
from collections import defaultdict
from warnings import catch_warnings, simplefilter, warn

import gsd
import gsd.hoomd
import numpy as np
from mbuild import Compound, clone
from mbuild.utils.io import run_from_ipython
from openbabel import pybel

from grits.utils import (
    comp_from_snapshot,
    get_bonds,
    has_common_member,
    has_number,
    snap_molecules,
)

__all__ = ["CG_Compound", "CG_System", "Bead"]


class CG_Compound(Compound):
    """Coarse-grained Compound.

    Wrapper for :py:class:`mbuild.Compound`. Coarse-grained mapping can be
    specified using SMARTS grammar.

    Parameters
    ----------
    compound : mbuild.Compound
        Fine-grain structure to be coarse-grained
    beads : dict, default None
        Dictionary with keys containing desired bead name and values containing
        SMARTS string specification of that bead. For example::

            beads = {"_B": "c1sccc1", "_S": "CCC"}

        would map a ``"_B"`` bead to any thiophene moiety (``"c1sccc1"``) found
        in the compound and an ``"_S"`` bead to a propyl moiety (``"CCC"``).
        User must provide only one of beads or mapping.
    mapping : dict or path, default None
        Either a dictionary or path to a json file of a dictionary. Dictionary
        keys contain desired bead name and SMARTS string specification of that
        bead and values containing list of tuples of atom indices::

            mapping = {"_B...c1sccc1"): [(0, 4, 3, 2, 1), ...]}

        User must provide only one of beads or mapping.
    allow_overlap : bool, default False,
        Whether to allow beads representing ring structures to share atoms.

    Attributes
    ----------
    atomistic : mbuild.Compound
        The atomistic structure.
    mapping : dict
        A mapping from atomistic to coarse-grain structure. Dictionary keys are
        the bead name and SMARTS string (separated by "..."), and the values are
        a list of tuples of fine-grain particle indices for each bead instance::

            {"_B...c1sccc1"): [(0, 4, 3, 2, 1), ...], ...}

    anchors : dict
        A mapping of the anchor particle indices in each bead. Dictionary keys
        are the bead name and the values are a set of indices::

            {"_B": {0, 2, 3}, ...}

    bond_map: list[tuple(str, tuple(int, int))]
        A list of the bond types and the anchors to use for that bond::

            [('_B-_S', (3, 0)), ...]
    """

    def __init__(
        self, compound, beads=None, mapping=None, allow_overlap=False, **kwargs
    ):
        super(CG_Compound, self).__init__(**kwargs)
        if (beads is None) == (mapping is None):
            raise ValueError(
                "Please provide only one of either beads or mapping."
            )
        self.atomistic = compound
        self.anchors = None
        self.bond_map = None

        if beads is not None:
            mol = compound.to_pybel()
            mol.OBMol.PerceiveBondOrders()

            self._set_mapping(beads, mol, allow_overlap)
        elif mapping is not None:
            if not isinstance(mapping, dict):
                with open(mapping, "r") as f:
                    mapping = json.load(f)
            self.mapping = mapping
        self._cg_particles()
        self._cg_bonds()

    def __repr__(self):
        """Format the CG_Compound representation."""
        return (
            f"<{self.name}: {self.n_particles} beads "
            + f"(from {self.atomistic.n_particles} atoms), "
            + "pos=({:.4f},{: .4f},{: .4f}), ".format(*self.pos)
            + f"{self.n_bonds:d} bonds>"
        )

    def _set_mapping(self, beads, mol, allow_overlap):
        """Set the mapping attribute."""
        matches = []
        for bead_name, smart_str in beads.items():
            smarts = pybel.Smarts(smart_str)
            if not smarts.findall(mol):
                warn(f"{smart_str} not found in compound!")
            for group in smarts.findall(mol):
                group = tuple(i - 1 for i in group)
                matches.append((group, smart_str, bead_name))

        seen = set()
        mapping = defaultdict(list)
        for group, smarts, name in matches:
            if allow_overlap:
                # smart strings for rings can share atoms
                # add bead regardless of whether it was seen
                if has_number(smarts):
                    seen.update(group)
                    mapping[f"{name}...{smarts}"].append(group)
                # alkyl chains should be exclusive
                else:
                    if has_common_member(seen, group):
                        pass
                    else:
                        seen.update(group)
                        mapping[f"{name}...{smarts}"].append(group)
            else:
                if has_common_member(seen, group):
                    pass
                else:
                    seen.update(group)
                    mapping[f"{name}...{smarts}"].append(group)

        n_atoms = mol.OBMol.NumHvyAtoms()
        if n_atoms != len(seen):
            warn("Some atoms have been left out of coarse-graining!")
            # TODO make this more informative
        self.mapping = mapping

    def _cg_particles(self):
        """Set the beads in the coarse-structure."""
        for key, inds in self.mapping.items():
            name, smarts = key.split("...")
            for group in inds:
                bead_xyz = self.atomistic.xyz[group, :]
                avg_xyz = np.mean(bead_xyz, axis=0)
                bead = Bead(name=name, pos=avg_xyz, smarts=smarts)
                self.add(bead)

    def _cg_bonds(self):
        """Set the bonds in the coarse structure."""
        bonds = get_bonds(self.atomistic)
        bead_inds = [
            (key.split("...")[0], group)
            for key, inds in self.mapping.items()
            for group in inds
        ]
        anchors = defaultdict(set)
        bond_map = []
        for i, (iname, igroup) in enumerate(bead_inds[:-1]):
            for j, (jname, jgroup) in enumerate(bead_inds[(i + 1) :]):
                for a, b in bonds:
                    if a in igroup and b in jgroup:
                        anchors[iname].add(igroup.index(a))
                        anchors[jname].add(jgroup.index(b))
                        bondinfo = (
                            f"{iname}-{jname}",
                            (igroup.index(a), jgroup.index(b)),
                        )

                    elif b in igroup and a in jgroup:
                        anchors[iname].add(igroup.index(b))
                        anchors[jname].add(jgroup.index(a))
                        bondinfo = (
                            f"{iname}-{jname}",
                            (igroup.index(b), jgroup.index(a)),
                        )
                    else:
                        continue
                    if bondinfo not in bond_map:
                        # If the bond is between two beads of the same type,
                        # add it to the end
                        if iname == jname:
                            bond_map.append(bondinfo)
                        # Otherwise add it to the beginning
                        else:
                            bond_map.insert(0, bondinfo)

                    self.add_bond([self[i], self[j + i + 1]])
        if anchors and bond_map:
            self.anchors = anchors
            self.bond_map = bond_map

    def save_mapping(self, filename=None):
        """Save the mapping operator to a json file.

        Parameters
        ----------
        filename : str, default None
            Filename where the mapping operator will be saved in json format.
            If None is provided, the filename will be CG_Compound.name +
            "_mapping.json".

        Returns
        -------
        str
            Path to saved mapping
        """
        if filename is None:
            filename = f"{self.name}_mapping.json"
        with open(filename, "w") as f:
            json.dump(self.mapping, f)
        print(f"Mapping saved to {filename}")
        return filename

    def visualize(
        self, show_ports=False, color_scheme={}, show_atomistic=False, scale=1.0
    ):  # pragma: no cover
        """Visualize the Compound using py3dmol.

        Allows for visualization of a Compound within a Jupyter Notebook.
        Modified to from :py:meth:`mbuild.Compound.visualize` to show atomistic
        elements (translucent) with larger CG beads.

        Parameters
        ----------
        show_ports : bool, default False
            Visualize Ports in addition to Particles
        color_scheme : dict, default {}
            Specify coloring for non-elemental particles keys are strings of
            the particle names values are strings of the colors::

                {'_CGBEAD': 'blue'}

        show_atomistic : bool, default False
            Show the atomistic structure stored in
            :py:attr:`CG_Compound.atomistic`
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
            # silence warning about No element found for CG bead
            with catch_warnings():
                simplefilter("ignore")
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

    Wrapper for :py:class:`mbuild.Compound`.

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
        super(Bead, self).__init__(element=None, **kwargs)


class CG_System:
    """Coarse-grained System.

    Coarse-grained mapping can be specified using SMARTS grammar or the indices
    of particles.

    Parameters
    ----------
    trajectory : path
        Path to a gsd file.
    beads : dict, default None
        Dictionary with keys containing desired bead name and values containing
        SMARTS string specification of that bead. For example::

            beads = {"_B": "c1sccc1", "_S": "CCC"}

        would map a ``"_B"`` bead to any thiophene moiety (``"c1sccc1"``) found
        in the compound and an ``"_S"`` bead to a propyl moiety (``"CCC"``).
        User must provide only one of beads or mapping.
    mapping : dict or path, default None
        Either a dictionary or path to a json file of a dictionary. Dictionary
        keys contain desired bead name and SMARTS string specification of that
        bead and values containing list of tuples of atom indices::

            mapping = {"_B...c1sccc1"): [(0, 4, 3, 2, 1), ...]}

        User must provide only one of beads or mapping.
    allow_overlap : bool, default False,
        Whether to allow beads representing ring structures to share atoms.

    Attributes
    ----------
    mapping : dict
        A mapping from atomistic to coarse-grain structure. Dictionary keys are
        the bead name and SMARTS string (separated by "..."), and the values are
        a list of tuples of fine-grain particle indices for each bead instance::

            {"_B...c1sccc1"): [(0, 4, 3, 2, 1), ...], ...}

    anchors : dict
        A mapping of the anchor particle indices in each bead. Dictionary keys
        are the bead name and the values are a set of indices::

            {"_B": {0, 2, 3}, ...}

    bond_map: list[tuple(str, tuple(int, int))]
        A list of the bond types and the anchors to use for that bond::

            [('_B-_S', (3, 0)), ...]
    """

    def __init__(
        self,
        trajectory,
        beads=None,
        mapping=None,
        allow_overlap=False,
        conversion_dict=None,
        scale=1.0,
    ):
        if (beads is None) == (mapping is None):
            raise ValueError(
                "Please provide only one of either beads or mapping."
            )
        self.trajectory = trajectory
        self.anchors = None
        self.bond_map = None
        self.conversion = conversion_dict
        self.scale = scale
        self.compounds = []

        if beads is not None:
            self._set_mapping(beads, allow_overlap)
        elif mapping is not None:
            if not isinstance(mapping, dict):
                with open(mapping, "r") as f:
                    mapping = json.load(f)
            self.mapping = mapping
        # self._cg_particles()
        # self._cg_bonds()

    def __repr__(self):
        """Format the CG_System representation."""
        return (
            f"<{self.name}: {self.n_particles} beads "
            + f"(from {self.atomistic.n_particles} atoms), "
            + "pos=({:.4f},{: .4f},{: .4f}), ".format(*self.pos)
            + f"{self.n_bonds:d} bonds>"
        )

    def _get_compounds(self):
        """Get compounds for each molecule type in the gsd trajectory."""
        # Use the first frame to find the coarse-grain mapping
        with gsd.hoomd.open(self.trajectory) as t:
            snap = t[0]

        # Break apart the snapshot into separate molecules
        molecules = snap_molecules(snap)
        mol_inds = []
        for i in range(max(molecules) + 1):
            mol_inds.append(np.where(molecules == i)[0])
        # If molecule length is different, it will be assumed to be different
        mol_lengths = [len(i) for i in mol_inds]
        uniq_mol_inds = []
        for l in set(mol_lengths):
            uniq_mol_inds.append(mol_inds[mol_lengths.index(l)])

        # Convert each unique molecule to a compound
        for inds in uniq_mol_inds:
            self.compounds.append(
                comp_from_snapshot(snap, inds, scale=self.scale)
            )

    def save_mapping(self, filename=None):
        """Save the mapping operator to a json file.

        Parameters
        ----------
        filename : str, default None
            Filename where the mapping operator will be saved in json format.
            If None is provided, the filename will be CG_Compound.name +
            "_mapping.json".

        Returns
        -------
        str
            Path to saved mapping
        """
        if filename is None:
            filename = f"{self.name}_mapping.json"
        with open(filename, "w") as f:
            json.dump(self.mapping, f)
        print(f"Mapping saved to {filename}")
        return filename

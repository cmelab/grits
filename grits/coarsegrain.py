"""GRiTS: Coarse-graining tools."""
import json
import os
import tempfile
from collections import defaultdict
from warnings import catch_warnings, simplefilter, warn

import freud
import gsd.hoomd
import numpy as np
from mbuild import Compound, clone
from mbuild.utils.io import run_from_ipython
from openbabel import pybel
from ele import element_from_symbol

from grits.utils import (
    NumpyEncoder,
    comp_from_snapshot,
    get_bonds,
    has_common_member,
    has_number,
    snap_molecules,
    get_major_axis,
    get_quaternion,
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

            mapping = {"_B...c1sccc1": [(0, 4, 3, 2, 1), ...]}

        User must provide only one of beads or mapping.
    allow_overlap : bool, default False
        Whether to allow beads representing ring structures to share atoms.
    add_hydrogens : bool, default False
        Whether to add hydrogens. Useful for united-atom models.
    aniso_beads : bool, default False
        Whether to calculate orientations for anisotropic beads.
        Note: Bead sizes should be fitted during paramaterization.
        Only Gay-Berne major axis orientations are calculated here.

    Attributes
    ----------
    atomistic : mbuild.Compound
        The atomistic structure.
    mapping : dict
        A mapping from atomistic to coarse-grain structure. Dictionary keys are
        the bead name and SMARTS string (separated by "..."), and the values are
        a list of tuples of fine-grain particle indices for each bead instance::

            {"_B...c1sccc1": [(0, 4, 3, 2, 1), ...], ...}

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
        compound,
        beads=None,
        mapping=None,
        allow_overlap=False,
        add_hydrogens=False,
        aniso_beads=False,
        **kwargs,
    ):
        super(CG_Compound, self).__init__(**kwargs)
        if (beads is None) == (mapping is None):
            raise ValueError(
                "Please provide only one of either beads or mapping."
            )
        self.atomistic = compound
        self.anchors = None
        self.bond_map = None
        self.aniso_beads = aniso_beads

        if beads is not None:
            mol = compound.to_pybel()
            mol.OBMol.PerceiveBondOrders()
            if add_hydrogens:
                n_atoms = mol.OBMol.NumAtoms()
                # This is a goofy work around necessary for the aromaticity
                # to be set correctly.
                with tempfile.NamedTemporaryFile() as f:
                    mol.write(format="mol2", filename=f.name, overwrite=True)
                    mol = list(pybel.readfile("mol2", f.name))[0]

                mol.addh()
                n_atoms2 = mol.OBMol.NumAtoms()
                print(f"Added {n_atoms2-n_atoms} hydrogens.")

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
                masses = np.array([self.atomistic[i].mass for i in group])
                tot_mass = sum(masses)
                bead_xyz = self.atomistic.xyz[group, :]
                avg_xyz = np.mean(bead_xyz, axis=0)
                orientation = None
                if self.aniso_beads:
                    # filter out hydrogens
                    hmass = element_from_symbol('H').mass
                    heavy_positions = bead_xyz[np.where(masses > hmass)]
                    major_axis, ab_idxs = get_major_axis(heavy_positions)
                    orientation = get_quaternion(major_axis)
                bead = Bead(name=name,
                            pos=avg_xyz,
                            smarts=smarts,
                            mass=tot_mass,
                            orientation=orientation)
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
    orientation : numpy array, default None
        Quaternion describing an anisotropic Gay-Berne
        bead's orientation.

    Attributes
    ----------
    smarts : str
        SMARTS string used to specify this Bead.
    orientation : numpy array
        Quaternion describing an anisotropic Gay-Berne
        bead's orientation.
    """

    def __init__(self, smarts=None, orientation=None, **kwargs):
        self.smarts = smarts
        self.orientation = orientation
        super(Bead, self).__init__(element=None, **kwargs)


class CG_System:
    """Coarse-grained System.

    Coarse-grained mapping can be specified using SMARTS grammar or the indices
    of particles.

    Parameters
    ----------
    gsdfile : path
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
        bead and values containing list of lists of atom indices::

            mapping = {"_B...c1sccc1": [[0, 4, 3, 2, 1], ...]}

        User must provide only one of beads or mapping.
        If a mapping is provided, the bonding between the beads will not be set.
    allow_overlap : bool, default False,
        Whether to allow beads representing ring structures to share atoms.
    conversion_dict : dictionary, default None
        Dictionary to map particle types to their element.
    scale : float, default 1.0
        Factor by which to scale length values.
    add_hydrogens : bool, default False
        Whether to add hydrogens. Useful for united-atom models.
    aniso_beads : bool, default False
        Whether to calculate orientations for anisotropic beads.
        Note: Bead sizes should be fitted during paramaterization.
        These are not calculated here.

    Attributes
    ----------
    mapping : dict
        A mapping from atomistic to coarse-grain structure. Dictionary keys are
        the bead name and SMARTS string (separated by "..."), and the values are
        a list of numpy arrays of fine-grain particle indices for each bead
        instance::

            {"_B...c1sccc1": [np.array([0, 4, 3, 2, 1]), ...], ...}
    """

    def __init__(
        self,
        gsdfile,
        beads=None,
        mapping=None,
        allow_overlap=False,
        conversion_dict=None,
        length_scale=1.0,
        mass_scale=1.0,
        add_hydrogens=False,
        aniso_beads=False,
    ):
        if (beads is None) == (mapping is None):
            raise ValueError(
                "Please provide only one of either beads or mapping."
            )
        self.gsdfile = gsdfile
        self._compounds = []
        self._inds = []
        self._bond_array = None

        if beads is not None:
            # get compounds
            self._get_compounds(
                beads=beads,
                allow_overlap=allow_overlap,
                length_scale=length_scale,
                mass_scale=mass_scale,
                conversion_dict=conversion_dict,
                add_hydrogens=add_hydrogens,
                aniso_beads=aniso_beads
            )

            # calculate the bead mappings for the entire trajectory
            self._set_mapping()
        elif mapping is not None:
            if not isinstance(mapping, dict):
                with open(mapping, "r") as f:
                    mapping = json.load(f)
            self.mapping = mapping

    def _get_compounds(
        self,
        beads,
        allow_overlap,
        length_scale,
        mass_scale,
        conversion_dict,
        add_hydrogens,
        aniso_beads

    ):
        """Get compounds for each molecule type in the gsd trajectory."""
        # Use the first frame to find the coarse-grain mapping
        with gsd.hoomd.open(self.gsdfile) as t:
            snap = t[0]

        # Use the conversion dictionary to map particle type to element symbol
        if conversion_dict is not None:
            snap.particles.types = [
                conversion_dict[i].symbol for i in snap.particles.types
            ]
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
            l = len(inds)
            mb_comp = comp_from_snapshot(
                    snapshot=snap,
                    indices=inds,
                    length_scale=length_scale,
                    mass_scale=mass_scale
            )
            self._compounds.append(
                CG_Compound(
                    compound=mb_comp,
                    beads=beads,
                    add_hydrogens=add_hydrogens,
                    aniso_beads=aniso_beads,
                )
            )
            self._inds.append(
                [mol_inds[i] for i in np.where(np.array(mol_lengths) == l)[0]]
            )

    def _set_mapping(self):
        """Scale the mapping from each compound to the entire trajectory."""

        def shift_value(i):
            n_before, n_bead = order[types[i]]
            return (
                n_comps * n_before
                + (i - n_before)
                + comp_idx * n_bead
                + bead_count
            )

        v_shift = np.vectorize(shift_value)

        self.mapping = {}
        all_bonds = []
        bead_count = 0
        for comp, inds in zip(self._compounds, self._inds):
            # Map particles
            for k, v in comp.mapping.items():
                self.mapping[k] = [i[list(g)] for i in inds for g in v]

            # Map bonds
            p = {p: i for i, p in enumerate(comp.particles())}
            bond_array = np.array(
                [
                    (p[i], p[j]) if p[i] < p[j] else (p[j], p[i])
                    for (i, j) in comp.bonds()
                ]
            )

            types = [p.name for p in comp.particles()]
            n_comps = len(inds)
            # Check that bond array exists
            if bond_array.size > 0:
                order = {
                    i: (types.index(i), types.count(i)) for i in set(types)
                }
                comp_bonds = []
                for comp_idx in range(n_comps):
                    comp_bonds.append(v_shift(bond_array))
                all_bonds += comp_bonds
            bead_count += n_comps * len(types)

        if all_bonds:
            all_bond_array = np.vstack(all_bonds)
            self._bond_array = all_bond_array[all_bond_array[:, 0].argsort()]

    def save_mapping(self, filename):
        """Save the mapping operator to a json file.

        Parameters
        ----------
        filename : str
            Filename where the mapping operator will be saved in json format.
        """
        with open(filename, "w") as f:
            json.dump(self.mapping, f, cls=NumpyEncoder)
        print(f"Mapping saved to {filename}")

    def save(self, cg_gsdfile, start=0, stop=None):
        """Save the coarse-grain system to a gsd file.

        Does not calculate the image of the coarse-grain bead.

        Retains:
            - configuration: box, step
            - particles: N, position, typeid, types
            - bonds: N, group, typeid, types

        Parameters
        ----------
        cg_gsdfile : str
            Filename for new gsd file. If file already exists, it will be
            overwritten.
        start : int, default 0
            Where to start reading the gsd trajectory the system was created
            with.
        stop : int, default None
            Where to stop reading the gsd trajectory the system was created
            with. If None, will stop at the last frame.
        """
        typeid = []
        types = [i.split("...")[0] for i in self.mapping]
        for i, inds in enumerate(self.mapping.values()):
            typeid.append(np.ones(len(inds)) * i)
        typeid = np.hstack(typeid)

        # Set up bond information if it exists
        bond_types = []
        bond_ids = []
        N_bonds = 0
        if self._bond_array is not None:
            N_bonds = self._bond_array.shape[0]
            for bond in self._bond_array:
                bond_pair = "-".join(
                    [
                        types[int(typeid[int(bond[0])])],
                        types[int(typeid[int(bond[1])])],
                    ]
                )
                if bond_pair not in bond_types:
                    bond_types.append(bond_pair)
                _id = np.where(np.array(bond_types) == bond_pair)[0]
                bond_ids.append(_id)

        with gsd.hoomd.open(cg_gsdfile, "wb") as new, gsd.hoomd.open(
            self.gsdfile, "rb"
        ) as old:
            if stop is None:
                stop = len(old)
            for s in old[start:stop]:
                new_snap = gsd.hoomd.Snapshot()
                position = []
                mass = []
                f_box = freud.Box.from_box(s.configuration.box)
                unwrap_pos = f_box.unwrap(
                    s.particles.position, s.particles.image
                )
                for i, inds in enumerate(self.mapping.values()):
                    position += [np.mean(unwrap_pos[x], axis=0) for x in inds]
                    mass += [sum(s.particles.mass[x]) for x in inds]

                position = np.vstack(position)
                images = f_box.get_images(position)
                position = f_box.wrap(position)

                new_snap.configuration.box = s.configuration.box
                new_snap.configuration.step = s.configuration.step
                new_snap.particles.N = len(typeid)
                new_snap.particles.position = position
                new_snap.particles.image = images
                new_snap.particles.typeid = typeid
                new_snap.particles.types = types
                new_snap.particles.mass = mass
                new_snap.bonds.N = N_bonds
                new_snap.bonds.group = self._bond_array
                new_snap.bonds.types = np.array(bond_types)
                new_snap.bonds.typeid = np.array(bond_ids)
                new.append(new_snap)

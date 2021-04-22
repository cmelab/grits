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

    @classmethod
    def from_gsd(cls, gsdfile, frame=-1, coords_only=False, scale=1.0):
        """
        Given a trajectory gsd file creates an CG_Compound.
        If there are multiple separate molecules, they are returned
        as one compound.

        Parameters
        ----------
        gsdfile : str, filename
        frame : int, frame number (default -1)
        coords_only : bool (default False)
            If True, return compound with no bonds
        scale : float, scaling factor multiplied to coordinates (default 1.0)

        Returns
        -------
        CG_Compound
        """
        with gsd.hoomd.open(gsdfile, "rb") as f:
            snap = f[frame]
        bond_array = snap.bonds.group
        n_atoms = snap.particles.N

        # Add particles
        comp = cls()
        comp.box = mb.box.Box(lengths=snap.configuration.box[:3] * scale)
        for i in range(n_atoms):
            name = snap.particles.types[snap.particles.typeid[i]]
            xyz = snap.particles.position[i] * scale
            charge = snap.particles.charge[i]

            atom = mb.Particle(name=name, pos=xyz, charge=charge)
            comp.add(atom, label=str(i))

        if not coords_only:
            # Add bonds
            for i in range(bond_array.shape[0]):
                atom1 = int(bond_array[i][0])
                atom2 = int(bond_array[i][1])
                comp.add_bond([comp[atom1], comp[atom2]])
        return comp

    def amber_to_element(self):
        """
        Pybel does not know how to parse atoms names in AMBER style
        so this functions renames them to their atomistic counterparts
        """
        for particle in self.particles():
            particle.name = amber_dict[particle.name]

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

    def from_pybel(pybel_mol, use_element=True):
        """
        Create a Compound from a Pybel.Molecule

        Parameters
        ---------
        pybel_mol: pybel.Molecule
        use_element : bool, default True
            If True, construct mb Particles based on the pybel Atom's element.
            If False, constructs mb Particles based on the pybel Atom's type

        Returns
        ------
        cmpd : CG_Compound
        """
        openbabel = import_("openbabel")
        cmpd = CG_Compound()
        resindex_to_cmpd = {}

        # Iterating through pybel_mol for atom/residue information
        # This could just as easily be implemented by
        # an OBMolAtomIter from the openbabel library,
        # but this seemed more convenient at time of writing
        # pybel atoms are 1-indexed, coordinates in Angstrom
        for atom in pybel_mol.atoms:
            xyz = np.array(atom.coords) / 10
            if use_element:
                try:
                    temp_name = Element[atom.atomicnum]
                except KeyError:
                    warn(
                        f"No element detected for atom at index {atom.idx} "
                        f"with number {atom.atomicnum}, type {atom.type}"
                    )
                    temp_name = atom.type
            else:
                temp_name = atom.type
            temp = mb.compound.Particle(name=temp_name, pos=xyz)
            # Is there a safer way to check for res?
            if hasattr(atom, "residue"):
                if atom.residue.idx not in resindex_to_cmpd:
                    res_cmpd = CG_Compound()
                    resindex_to_cmpd[atom.residue.idx] = res_cmpd
                    cmpd.add(res_cmpd)
                resindex_to_cmpd[atom.residue.idx].add(temp)
            else:
                cmpd.add(temp)

        # Iterating through pybel_mol.OBMol for bond information
        # Bonds are 0-indexed, but the atoms are 1-indexed
        # Bond information doesn't appear stored in pybel_mol,
        # so we need to look into the OBMol object,
        # using an iterator from the openbabel library
        for bond in openbabel.OBMolBondIter(pybel_mol.OBMol):
            cmpd.add_bond(
                [cmpd[bond.GetBeginAtomIdx() - 1], cmpd[bond.GetEndAtomIdx() - 1]]
            )

        if hasattr(pybel_mol, "unitcell"):
            box = mb.box.Box(
                lengths=[
                    pybel_mol.unitcell.GetA() / 10,
                    pybel_mol.unitcell.GetB() / 10,
                    pybel_mol.unitcell.GetC() / 10,
                ],
                angles=[
                    pybel_mol.unitcell.GetAlpha(),
                    pybel_mol.unitcell.GetBeta(),
                    pybel_mol.unitcell.GetGamma(),
                ],
            )
            cmpd.periodicity = box.lengths
        else:
            warn("No unitcell detected for pybel.Molecule {}".format(pybel_mol))
            box = None

        cmpd.box = box

        return cmpd

    def wrap(self):
        """
        Finds particles which are out of the box and translates
        them to within the box.
        """
        try:
            freud_box = mb_to_freud_box(self.box)
        except TypeError:
            print("Can't wrap because CG_Compound.box values aren't assigned.")
            return
        particles = [part for part in self.particles()]
        # find rows where particles are out of the box
        for row in np.argwhere(abs(self.xyz) > self.box.maxs / 2)[:, 0]:
            new_xyz = freud_box.wrap(particles[row].pos)
            particles[row].translate_to(new_xyz)

    def unwrap(self, d_tolerance=0.22, _count=0):
        """
        Used to correct molecules which span the periodic boundary by translating particles
        to their real-space position. The function uses a distance tolerance to detect
        bonds which span the periodic boundary and from those determines which particle
        should be considered an outlier, finds other particles the outlier is bonded to,
        and shifts their position.

        Parameters
        ----------
        d_tolerance = float, distance beyond which a bond is considered "bad" (default=0.22)
        _count = int, used in recursive algorithm to prevent function from getting stuck
                 fixing bonds which span the pbc -- user should not set this value.

        if function is getting stuck in an endless loop, try adjusting d_tolerance
        """

        molecules = self.get_molecules()
        particles = [part for part in self.particles()]

        def check_bad_bonds(compound):
            """
            Used for identifying particles whose bonds span the periodic boundary.
            Finds particles indices in the compound with bonds longer than the
            distance tolerance.

            Parameters
            ----------
            compound : CG_Compound

            Returns
            -------
            list of tuples of particle indices
            """
            bad_bonds = [
                bond
                for bond in compound.bonds()
                if distance(bond[0].pos, bond[1].pos) > d_tolerance
            ]
            maybe_outliers = [
                (particles.index(bond[0]), particles.index(bond[1]))
                for bond in bad_bonds
            ]
            return maybe_outliers

        maybe_outliers = check_bad_bonds(self)
        if not maybe_outliers:
            print(
                f"No bonds found longer than {d_tolerance}. Either compound doesn't need"
                + " unwrapping or d_tolerance is too small. No changes made."
            )
            return

        def find_outliers(compound):
            """
            Finds "outliers" (bonded particles which span the periodic boundary).
            Starts by finding bonds that are too long, then determines which particle
            in the pair is an outlier based on whether removal of that particle reduces
            the average distance from the particles in the molecule to the geometric center.
            From these, the function follows the bond graph and adds all particles
            bonded to the outliers.

            Parameters
            ----------
            compound : CG_Compound

            Returns
            -------
            set of the particle indices of all outliers in the compound.
            """

            def _is_outlier(index):
                for molecule in molecules:
                    if index in molecule:
                        test_molecule = molecule.copy()
                        test_molecule.remove(index)
                        a = compound.xyz[list(molecule), :]
                        b = compound.xyz[list(test_molecule), :]
                        center_a = np.mean(a, axis=0)
                        center_b = np.mean(b, axis=0)
                        avg_dist_a = np.mean(v_distance(a, center_a))
                        avg_dist_b = np.mean(v_distance(b, center_b))
                return avg_dist_a > avg_dist_b

            def _d_to_center(index):
                for molecule in molecules:
                    if index in molecule:
                        mol_xyz = compound.xyz[list(molecule), :]
                        center = np.mean(mol_xyz, axis=0)
                        dist = distance(particles[index].pos, center)
                return dist

            outliers = set()
            checked = set()
            for tup in maybe_outliers:
                if _is_outlier(tup[0]) and _is_outlier(tup[1]):
                    # add whichever is further from the center
                    d_0 = _d_to_center(tup[0])
                    d_1 = _d_to_center(tup[1])
                    if d_0 > d_1:
                        outliers.add(tup[0])
                    elif d_1 > d_0:
                        outliers.add(tup[1])
                    else:
                        raise RuntimeError(
                            f"Can't determine which is outlier between indices {tup}"
                        )
                elif _is_outlier(tup[0]):
                    outliers.add(tup[0])
                elif _is_outlier(tup[1]):
                    outliers.add(tup[1])
                checked.add(tup[0])
                checked.add(tup[1])

            starts = outliers.copy()
            while starts:
                index = starts.pop()
                outliers.update(bond_dict[index] - checked)
                starts.update(bond_dict[index] - checked)
                checked.add(index)
            return outliers

        bond_dict = self.bond_dict()
        outliers = find_outliers(self)

        # organize outliers by molecule
        outlier_dict = defaultdict(set)
        for outlier in outliers:
            mol_ind = [i for i, mol in enumerate(molecules) if outlier in mol]
            if mol_ind:
                outlier_dict[mol_ind[0]].add(outlier)

        # find the center of the molecule without outliers
        for mol_ind in outlier_dict:
            molecule = molecules[mol_ind].copy()
            molecule -= outlier_dict[mol_ind]  # not outlier indices
            mol_xyz = self.xyz[list(molecule), :]
            mol_avg = np.mean(mol_xyz, axis=0)

            # translate the outlier to its real-space position found using
            # freud.box.unwrap. the direction is determined using the
            # difference between the particle position and the molecule center
            freud_box = mb_to_freud_box(self.box)
            for outlier in outlier_dict[mol_ind]:
                image = mol_avg - particles[outlier].pos
                img = np.where(image > self.box.maxs / 2, 1, 0) + np.where(
                    image < -self.box.maxs / 2, -1, 0
                )
                new_xyz = freud_box.unwrap(particles[outlier].pos, img)
                particles[outlier].translate_to(new_xyz)

        # check if any bad bonds remain
        bad_bonds = check_bad_bonds(self)
        if bad_bonds:
            if _count < 5:
                _count += 1
                print(f"Bad bonds still present! Trying unwrap again. {_count}")
                self.unwrap(d_tolerance=d_tolerance, _count=_count)
            else:
                print("Bad bonds still present and try limit exceeded.")

    def bond_dict(self):
        """
        given an CG_Compound return an dict of the particle indices for each bond

        CG_Compound.bond_dict() --> dict of sets
        """
        parts = [part for part in self.particles()]
        bond_dict = defaultdict(set)
        for bond in self.bonds():
            bond_dict[int(parts.index(bond[0]))].add(int(parts.index(bond[1])))
            bond_dict[int(parts.index(bond[1]))].add(int(parts.index(bond[0])))
        return bond_dict

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

    def find_angles(self):
        """
        Adapted from cme_utils.manip.builder.building_block.find_angles()
        Finds unique angle constraints and their types.

        Returns
        -------
        Dictionary with keys correponding to the angle types and
        values which list the particle indices which have this angle type
        """
        angles = []
        bond_dict = self.bond_dict()
        for i in range(self.n_particles):
            for n1 in bond_dict[i]:
                for n2 in bond_dict[n1]:
                    if n2 != i:
                        if n2 > i:
                            angles.append((i, n1, n2))
                        else:
                            angles.append((n2, n1, i))
        angles = sorted(set(angles))
        angle_types = []
        for t in angles:
            angle_types.append(self.tuple_to_names(t))

        angle_dict = defaultdict(list)
        for a, b in zip(angle_types, angles):
            angle_dict[a].append(b)
        return angle_dict

    def find_bonds(self):
        """
        Finds unique bond constraints and their types.

        Returns
        -------
        Dictionary with keys correponding to the bond types and
        values which list the particle indices which have this type
        """
        bonds = []
        bond_dict = self.bond_dict()
        for i in range(self.n_particles):
            for n1 in bond_dict[i]:
                if n1 > i:
                    bonds.append((i, n1))
                else:
                    bonds.append((n1, i))
        bonds = sorted(set(bonds))
        bond_types = []
        for t in bonds:
            bond_types.append(self.tuple_to_names(t))

        bond_dict = defaultdict(list)
        for a, b in zip(bond_types, bonds):
            bond_dict[a].append(b)
        return bond_dict

    def find_pairs(self):
        """
        Finds unique (coarse-grained) pair types
        (coarse particle names start with "_")

        Returns
        -------
        list of tuples of pair names
        """
        particles = {p.name for p in self.particles() if p.name[0] == "_"}
        pairs = set()
        for i in particles:
            for j in particles:
                pair = tuple(sorted([i, j]))
                pairs.add(pair)
        return sorted(pairs)


    def is_bad_bond(self, tup):
        """
        Determines whether a bond spans the periodic boundary based on a distance
        cutoff of the self.box.maxs/2

        Parameters
        ----------
        tup : tuple, indices of the bonded particles

        Returns
        -------
        bool
        """
        if tup not in self.get_bonds() and tup[::-1] not in self.get_bonds():
            print(f"Bond {tup} not found in compound! Aborting...")
            return
        pair = [p for i, p in enumerate(self.particles()) if i == tup[0] or i == tup[1]]
        test = np.where(abs(pair[0].xyz - pair[1].xyz) > self.box.maxs / 2)[1]
        if test.size > 0:
            return True
        else:
            return False

    def unwrap_position(self, tup):
        """
        Given the indices of a bonded pair which spans the periodic boundary,
        moves the second index to it's real-space position.

        Parameters
        ----------
        tup : tuple, indices (2) of bonded particles

        Returns
        -------
        np.ndarray(3,), unwrapped coordinates for index in tup[1]
        (if you want to move the first index, enter it as tup[::-1])
        """
        freud_box = mb_to_freud_box(self.box)
        pair = [p for i, p in enumerate(self.particles()) if i == tup[0] or i == tup[1]]
        diff = pair[0].pos - pair[1].pos
        img = np.where(diff > self.box.maxs / 2, 1, 0) + np.where(
            diff < -self.box.maxs / 2, -1, 0
        )
        return freud_box.unwrap(pair[1].pos, img)


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


amber_dict = {
    "c": "C",
    "c1": "C",
    "c2": "C",
    "c3": "C",
    "ca": "C",
    "cp": "C",
    "cq": "C",
    "cc": "C",
    "cd": "C",
    "ce": "C",
    "cf": "C",
    "cg": "C",
    "ch": "C",
    "cx": "C",
    "cy": "C",
    "cu": "C",
    "cv": "C",
    "h1": "H",
    "h2": "H",
    "h3": "H",
    "h4": "H",
    "h5": "H",
    "ha": "H",
    "hc": "H",
    "hn": "H",
    "ho": "H",
    "hp": "H",
    "hs": "H",
    "hw": "H",
    "hx": "H",
    "f": "F",
    "cl": "Cl",
    "br": "Br",
    "i": "I",
    "n": "N",
    "n1": "N",
    "n2": "N",
    "n3": "N",
    "n4": "N",
    "na": "N",
    "nb": "N",
    "nc": "N",
    "nd": "N",
    "ne": "N",
    "nf": "N",
    "nh": "N",
    "no": "N",
    "o": "O",
    "oh": "O",
    "os": "O",
    "ow": "O",
    "p2": "P",
    "p3": "P",
    "p4": "P",
    "p5": "P",
    "pb": "P",
    "pc": "P",
    "pd": "P",
    "pe": "P",
    "pf": "P",
    "px": "P",
    "py": "P",
    "s": "S",
    "s2": "S",
    "s4": "S",
    "s6": "S",
    "sh": "S",
    "ss": "S",
    "sx": "S",
    "sy": "S",
}


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

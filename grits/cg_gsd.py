from cmeutils import gsd_utils
from cmeutils.gsd_utils import snap_molecule_cluster
import freud
import numpy as np

class System:
    """
    """
    def __init__(self,
            atoms_per_monomer,
            gsd_file=None,
            snap=None,
            gsd_frame=-1):
        self.atoms_per_monomer = atoms_per_monomer
        self.snap = gsd_utils._validate_inputs(gsd_file, snap, gsd_frame)
        self.clusters = snap_molecule_cluster(snap=self.snap)
        self.molecule_ids = set(self.clusters)
        self.n_molecules = len(self.molecule_ids)
        self.n_atoms = len(self.clusters)
        self.n_monomers = int(self.n_atoms / self.atoms_per_monomer)
        self.molecules = [Molecule(self, i) for i in self.molecule_ids] 
        self.box = gsd_utils.snap_box(gsd_file, snap, gsd_frame)
        assert len(self.molecules) == self.n_molecules

    def monomers(self):
        """Generate all of the monomers from each molecule
        in System.molecules.

        Yields:
        -------
        polymers.Monomer
     
        """
        for molecule in self.molecules:
            for monomer in molecule.monomers:
                yield monomer

    def segments(self):
        """Generate all of the segments from each molecule
        in System.

        Yields:
        -------
        polymers.Segment

        """
        for molecule in self.molecules:
            for segment in molecule.segments:
                yield segment

    def components(self):
        """Generate all of the components from each molecule in System.

        Yields:
        -------
        polymers.Component

        """
        for monomer in self.monomers():
            for component in monomer.components:
                yield component


class Structure:
    """Base class for the Molecule(), Segment(), and Monomer() classes.

    Parameters:
    -----------
    system : 'cmeutils.polymers.System', required
        The system object initially created from the input .gsd file.
    atom_indices : np.ndarray(n, 3), optional, default=None
        The atom indices in the system that belong to this specific structure.
    molecule_id : int, optional, default=None
        The ID number of the specific molecule from system.molecule_ids.

    Attributes:
    -----------
    system : 'cmeutils.polymers.System'
        The system that this structure belong to. Contains needed information
        about the box, and gsd snapshot which are used elsewhere.
    atom_indices : np.ndarray(n_atoms, 3)
        The atom indices in the system that belong to this specific structure
    n_atoms : int
        The number of atoms that belong to this specific structure
    atom_positions : np.narray(n_atoms, 3)
        The x, y, z coordinates of the atoms belonging to this structure.
        The positions are wrapped inside the system's box.
    center_of_mass : np.1darray(1, 3)
        The x, y, z coordinates of the structure's center of mass.

    """
    def __init__(self,
            system,
            atom_indices=None,
            name=None,
            parent=None,
            molecule_id=None
            ):
        self.system = system
        self.name = name
        self.parent = parent
        if molecule_id != None:
            self.atom_indices = np.where(self.system.clusters == molecule_id)[0]
            self.molecule_id = molecule_id
        else:
            self.atom_indices = atom_indices
        self.n_atoms = len(self.atom_indices)

    def generate_monomers(self):
        if isinstance(self, Monomer):
            return self
        structure_length = int(self.n_atoms / self.system.atoms_per_monomer)
        monomer_indices = np.array_split(self.atom_indices, structure_length)
        assert len(monomer_indices) == structure_length
        return [Monomer(self, i) for i in monomer_indices]

    @property
    def atom_positions(self):
        """The wrapped coordinates of every particle in the structure
        as they exist in the periodic box.

        Returns:
        --------
        numpy.ndarray, shape=(n, 3), dtype=float

        """
        return self.system.snap.particles.position[self.atom_indices]

    @property
    def center(self):
        """The (x,y,z) position of the center of the structure accounting
        for the periodic boundaries in the system.
        
        Returns:
        --------
        numpy.ndarray, shape=(3,), dtype=float

        """
        freud_box = freud.Box(
                Lx = self.system.box[0],
                Ly = self.system.box[1],
                Lz = self.system.box[2]
                )
        return freud_box.center_of_mass(self.atom_positions)

    @property 
    def unwrapped_atom_positions(self):
        """The unwrapped coordiantes of every particle in the structure.
        The positions are unwrapped using the images for each particle
        stored in the hoomd snapshot.

        Returns:
        --------
        numpy.ndarray, shape=(n, 3), dtype=float

        """
        images = self.system.snap.particles.image[self.atom_indices]
        return self.atom_positions + (images * self.system.box[:3]) 

    @property
    def unwrapped_center(self):
        """The (x,y,z) position of the center using the structure's
        unwrapped coordiantes.

        Returns:
        --------
        numpy.ndarray, shape=(3,), dtype=float

        """
        x_mean = np.mean(self.unwrapped_atom_positions[:,0])
        y_mean = np.mean(self.unwrapped_atom_positions[:,1])
        z_mean = np.mean(self.unwrapped_atom_positions[:,2])
        return np.array([x_mean, y_mean, z_mean])

class Molecule(Structure):
    """
    """
    def __init__(self, system, molecule_id):
        super(Molecule, self).__init__(
                system=system,
                molecule_id=molecule_id
                )
        self.monomers = self.generate_monomers() 
        self.n_monomers = len(self.monomers)
        self.segments = None
        self.n_segments = None
        self.sequence = None

    def assign_types(self, sequence):
        n = self.n_monomers // len(sequence)
        monomer_sequence = sequence * n
        monomer_sequence += sequence[:(self.n_monomers - len(monomer_sequence))]
        for i, name in enumerate(list(monomer_sequence)):
            self.monomers[i].name = name

    def generate_segments(self, monomers_per_segment):
        """
        Creates a `Segment` that contains a subset of it's `Molecule` atoms.

        Segments are defined as containing a certain number of monomers.
        For example, if you want 3 subsequent monomers contained in a single
        Segment instance, use `monomers_per_segment = 3`.
        The segments are accessible in the `Molecule.segments` attribute.

        Parameters:
        -----------
        monomers_per_segment : int, required
            Define the number of consecutive monomers that belong to
            each segment.

        """
        segments_per_molecule = int(self.n_monomers / monomers_per_segment)
        segment_indices = np.array_split(
                self.atom_indices,
                segments_per_molecule
                )
        self.segments = [Segment(self, i) for i in segment_indices]
        self.n_segments = len(self.segments)
    

class Monomer(Structure):
    """
    """
    def __init__(self, parent, atom_indices):
        super(Monomer, self).__init__(
                system=parent.system,
                parent=parent,
                atom_indices=atom_indices
                )
        self.components = None
        assert self.n_atoms == self.system.atoms_per_monomer 
        
    def generate_components(self, index_mapping):
        components = []
        for name, indices in index_mapping.items():
            if all([isinstance(i, list) for i in indices]):
                for i in indices:
                    component = Component(
                            monomer=self,
                            name=name,
                            atom_indices = self.atom_indices[i]
                            )
                    components.append(component)
            else:
                component = Component(
                        monomer=self,
                        name=name,
                        atom_indices = self.atom_indices[indices]
                        )
                components.append(component)
        self.components = components


class Segment(Structure):
    """
    """
    def __init__(self, molecule, atom_indices):
        super(Segment, self).__init__(
                system=molecule.system,
                atom_indices=atom_indices,
                parent = molecule
                )
        self.monomers = self.generate_monomers()
        assert len(self.monomers) ==  int(
                self.n_atoms / self.system.atoms_per_monomer
                )

class Component(Structure):
    def __init__(self, monomer, atom_indices, name):
        super(Component, self).__init__(
                system=monomer.system,
                parent=monomer.parent,
                atom_indices=atom_indices,
                name=name
                )
        self.monomer = monomer
        

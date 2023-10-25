from base_test import BaseTest


def test_num2str():
    from grits.utils import num2str

    assert num2str(0) == "A"
    assert num2str(25) == "Z"
    assert num2str(26) == "AA"


def test_get_hydrogen():
    from mbuild import Compound, load

    from grits.utils import get_hydrogen

    cl2 = load("[Cl][Cl]", smiles=True)
    assert get_hydrogen(cl2, cl2[0]) is None

    methane = load("C", smiles=True)
    assert isinstance(get_hydrogen(methane, methane[0]), Compound)


class TestAnisoUtils(BaseTest):
    def test_get_heavy_atoms(self, butane_gsd):
        import gsd.hoomd

        from grits.utils import get_heavy_atoms

        with butane_gsd as f:
            assert type(f) == gsd.hoomd.HOOMDTrajectory
            frame = f[0]
            particles = frame.particles
            heavy_positions, heavy_masses = get_heavy_atoms(particles)
            # 4 carbons -> 4 entries
            assert len(heavy_masses) == len(heavy_positions) == 4
            # approximate mass
            assert round(sum(heavy_masses)) == 48

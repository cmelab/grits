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

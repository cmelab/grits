from os import path
import pytest


test_dir = path.dirname(__file__)


class BaseTest:
    @pytest.fixture
    def p3ht(self):
        import mb

        p3ht = mb.load(path.join(test_dir, "assets/P3HT_16.mol2"))
        return p3ht

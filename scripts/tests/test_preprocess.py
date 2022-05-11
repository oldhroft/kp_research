from ..helpers.preprocess import *

def test_categorize():
    assert categorize(40) == 1
    assert categorize(70) == 2
    assert categorize(30) == 0

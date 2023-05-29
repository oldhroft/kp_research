import os

from pandas import read_csv

from scripts.pipeline.preprocess import *
from scripts.helpers.yaml_utils import load_yaml


LOCATION = os.path.dirname(os.path.realpath(__file__))


def test_categorize():
    
    borders = [20, 40]
    assert categorize(2, borders) == 0
    assert categorize(30, borders) == 1
    assert categorize(50, borders) == 2
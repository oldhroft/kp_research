from pandas import read_csv

from ..helpers.preprocess import *
from ..helpers.yaml_utils import load_yaml

def test_categorize():
    assert categorize(40) == 1
    assert categorize(70) == 2
    assert categorize(30) == 0

DF = read_csv('scripts/tests/test_data/test.csv')

def test_preprocess_3h():
    conf = load_yaml('scripts/tests/test_yamls/test_vars.yaml')
    df = DF.pipe(preprocess_3h)[conf['variables']]
    assert df.isna().sum().sum() == 0, 'NA values are present'
    assert df.shape[0] == DF.shape[0] // 3 + 1, 'Shape is not preserved'
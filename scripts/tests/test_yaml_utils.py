
from ..helpers.yaml_utils import *

def test_simple_yaml_load():
    data = load_yaml("scripts/tests/test_yamls/test1.yaml")
    assert data['test']['value1'] == 1, "Yaml file is not correct"

def test_include_yaml():
    data = load_yaml("scripts/tests/test_yamls/test2.yaml")
    assert data['test']['import']['test']['value1'] == 1, "Yaml !include option does not work"
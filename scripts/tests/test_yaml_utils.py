import os
from ..helpers.yaml_utils import *


class TestYamlLoad:
    def test_simple_yaml_load(self):
        data = load_yaml("scripts/tests/test_yamls/test1.yaml")
        assert data["test"]["value1"] == 1, "Yaml file is not correct"

    def test_include_yaml(self):
        data = load_yaml("scripts/tests/test_yamls/test2.yaml")
        assert (
            data["test"]["import"]["test"]["value1"] == 1
        ), "Yaml !include option does not work"

    def test_environ_yaml(self):
        os.environ["KEY"] = "value"
        data = load_yaml("scripts/tests/test_yamls/test_env.yaml")
        assert data["test"]["key"] == "value"

    def test_complex_environ_yaml(self):
        os.environ["KEY"] = "2"
        data = load_yaml("scripts/tests/test_yamls/test_env.yaml")
        assert data["test"]["key"] == 2

    def test_complex_environ_v1_yaml(self):
        os.environ["KEY"] = "'folder'"
        data = load_yaml("scripts/tests/test_yamls/test_env.yaml")
        assert data["test"]["key"] == "folder"

    def test_complex_environ_v2_yaml(self):
        os.environ["KEY"] = "[1, 2, 3]"
        data = load_yaml("scripts/tests/test_yamls/test_env.yaml")
        assert str(data["test"]["key"]) == str([1, 2, 3])
        assert isinstance(data["test"]["key"], list)


def test_dict_to_yaml_str():
    test_dct = {"key": "value"}
    answer = "\nkey: value\n"
    string = dict_to_yaml_str(test_dct)
    assert string == answer

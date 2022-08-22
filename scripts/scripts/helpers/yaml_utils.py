import yaml
import os
import yaml
from typing import Any, IO


class Loader(yaml.SafeLoader):  # pragma: no cover
    """YAML Loader with `!include` constructor."""

    def __init__(self, stream: IO) -> None:
        """Initialise Loader."""

        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir

        super().__init__(stream)


def construct_include(loader: Loader, node: yaml.Node) -> Any:  # pragma: no cover
    """Include file referenced at node."""

    filename = os.path.abspath(
        os.path.join(loader._root, loader.construct_scalar(node))
    )
    extension = os.path.splitext(filename)[1].lstrip(".")

    with open(filename, "r") as f:
        if extension in ("yaml", "yml"):
            return yaml.load(f, Loader)
        else:
            return "".join(f.readlines())


import re

path_matcher = re.compile(r"\$\{([^}^{]+)\}")

from ast import literal_eval


def path_constructor(loader, node):  # pragma: no cover
    """Extract the matched value, expand env variable, and replace the match"""
    value = node.value
    match = path_matcher.match(value)
    for item in path_matcher.findall(value):
        value = re.sub(path_matcher, os.environ.get(item), value, count=1)
        try:
            value = literal_eval(value)
        except Exception as e:
            pass

    return value


def load_yaml(file: str) -> dict:

    yaml.add_implicit_resolver("!path", path_matcher, None, Loader)  # pragma: no cover
    yaml.add_constructor("!path", path_constructor, Loader)  # pragma: no cover
    yaml.add_constructor("!include", construct_include, Loader)  # pragma: no cover

    with open(
        file,
        "r",
        encoding="utf-8",
    ) as file:
        data = yaml.load(file, Loader)
    return data


def dict_to_yaml_str(obj: dict) -> str:
    return "\n" + yaml.dump(obj)

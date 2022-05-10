import yaml
import os
import yaml
from typing import Any, IO

class Loader(yaml.SafeLoader): #pragma: no cover
    """YAML Loader with `!include` constructor."""

    def __init__(self, stream: IO) -> None:
        """Initialise Loader."""

        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir

        super().__init__(stream)

def construct_include(loader: Loader, node: yaml.Node) -> Any:
    """Include file referenced at node."""

    filename = os.path.abspath(os.path.join(loader._root, loader.construct_scalar(node)))
    extension = os.path.splitext(filename)[1].lstrip('.')

    with open(filename, 'r') as f:
        if extension in ('yaml', 'yml'):
            return yaml.load(f, Loader)
        else:
            return ''.join(f.readlines())

yaml.add_constructor('!include', construct_include, Loader) # pragma: no cover

def load_yaml(file: str) -> dict:

    with open(file, 'r', encoding='utf-8', ) as file:
        data = yaml.load(file, Loader)
    
    return data
    
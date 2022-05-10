from ..helpers.utils import *
import os

def test_create_folder():
    if os.path.exists('scripts/tests/test1'):
        os.rmdir('scripts/tests/test1')
    create_folder('scripts/tests/test1')
    flag = os.path.exists('scripts/tests/test1')
    os.rmdir('scripts/tests/test1')
    assert flag, "Folder scripts/tests/test1 not created"


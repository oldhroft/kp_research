from ..helpers.utils import *
from ..helpers.utils import _choose_suffix_name
import os

def test_create_folder():
    if os.path.exists('scripts/tests/test1'):
        os.rmdir('scripts/tests/test1')
    create_folder('scripts/tests/test1')
    flag = os.path.exists('scripts/tests/test1')
    os.rmdir('scripts/tests/test1')
    assert flag, "Folder scripts/tests/test1 not created"

def test__choose_suffix_name():
    assert _choose_suffix_name(True, 'future') == 'future', 'Unexpected suffix name'
    assert _choose_suffix_name(False, None) == 'lag', 'Unexpected suffix name'
    assert _choose_suffix_name(True, None) == 'lead', 'Unexpected suffix name'
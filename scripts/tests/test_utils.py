from random import sample
import pytest

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

def test_class_decorator():

    def function_decorator(func):
        def new_function(cls_instance, x):
            return func(cls_instance, x) + 1
        new_function.__name__ = func.__name__
        return new_function

    @decorate_class(function_decorator)
    class SampleClass:
        def __init__(self) -> None:
            pass

        def some_function(self, x):
            return x
    cls_instance = SampleClass()

    assert cls_instance.some_function(1) == 2
    assert hasattr(SampleClass, "__init__")

def test_class_decorator_with_static_method():

    @decorate_class(staticmethod)
    class SampleClass:
        def __init__(self) -> None:
            pass

        def some_function(x):
            return x

    assert hasattr(SampleClass, "__init__")
    assert SampleClass.some_function(1) == 1


def test_add_to_environ():
    sample_conf = [
        "folder=test_folder",
        "variable=2"
    ]

    add_to_environ(sample_conf)
    assert os.environ["folder"] == 'test_folder'
    assert os.environ['variable'] == "2"

def test__choose_suffix_name():
    assert _choose_suffix_name(True, 'future') == 'future', 'Unexpected suffix name'
    assert _choose_suffix_name(False, None) == 'lag', 'Unexpected suffix name'
    assert _choose_suffix_name(True, None) == 'lead', 'Unexpected suffix name'

from ..helpers.utils import _trim
from pandas import read_csv

DF = read_csv('scripts/tests/test_data/test.csv')

class Test_Trim:

    def test_no_trim(self):
        s = _trim(DF, forward=True, trim=False, lags=10100000)
        assert s.shape == DF.shape
    
    def test_forward_trim(self):
        s = _trim(DF, forward=True, trim=True, lags=3)
        assert s.shape == (DF.shape[0] - 3, DF.shape[1])
        
        last1 = DF.iloc[-4]['Kp*10']
        last2 = s.iloc[-1]['Kp*10']
        assert last1 == last2
    
    def test_backward_trim(self):
        s = _trim(DF, forward=False, trim=True, lags=3)
        assert s.shape == (DF.shape[0] - 3, DF.shape[1])
        
        last1 = DF.iloc[3]['Kp*10']
        last2 = s.iloc[0]['Kp*10']
        assert last1 == last2


class TestAddLags:

    def test_forward_one_column(self):
        s = add_lags(DF, 'Kp*10', forward=True, lags=2, trim=True, return_cols=False)
        assert 'Kp*10_lead_1' in s.columns and 'Kp*10_lead_2' in s.columns
        assert (s['Kp*10_lead_2'].values == DF.iloc[2: ]['Kp*10'].values).all()
    
    def test_backward_one_column(self):
        s = add_lags(DF, 'Kp*10', forward=False, lags=2, trim=True, return_cols=False)
        assert 'Kp*10_lag_1' in s.columns and 'Kp*10_lag_2' in s.columns
        assert (s['Kp*10_lag_2'].values == DF.iloc[:-2]['Kp*10'].values).all()

    def test_backward_two_column(self):
        s, columns = add_lags(DF, ['dttm', 'Kp*10'], forward=False, 
                          lags=1, trim=True, return_cols=True)

        assert s.shape == (DF.shape[0] - 1, DF.shape[1] + 2)
        assert 'dttm_lag_1' in columns and 'Kp*10_lag_1' in columns and len(columns) == 2
    
    def test_negative_lags(self):
        with pytest.raises(ValueError):
            add_lags(DF, ['dttm', 'Kp*10'], forward=False, 
                     lags=-1, trim=True, return_cols=True)

    def test_str_lags(self):
        with pytest.raises(ValueError):
            add_lags(DF, ['dttm', 'Kp*10'], forward=False, 
                     lags='str', trim=True, return_cols=True)
    
    def test_default_subset(self):
        df_new = add_lags(DF, subset=None, forward=False, 
                          lags=1, trim=True, return_cols=False)
        
        assert df_new.shape[1] == 2 * DF.shape[1]
    
    def test_no_lags(self):
        df_new = add_lags(DF, subset=None, forward=False, 
                          lags=0, trim=True, return_cols=False)
        assert df_new.shape == DF.shape


from sklearn.metrics import f1_score
from numpy import array, sqrt

def test_columnwise_score():

    '''
    first_column:
        0: Precision: 0.5, Recall: 1., F1: 2 / 3
        1: Precision: 1, Recall: 0,5, F1: 2 / 3
        macro: 2 / 3s
    second_column:
        0: F1: 1
        1: F1: 1
        macro: 1
    '''

    y_true = [
        [1, 0],
        [1, 1],
        [0, 1]
    ]

    y_pred = [
        [0, 0],
        [1, 1],
        [0, 1]
    ]

    true_scores = array([2 / 3, 1.])
    score = columnwise_score(f1_score, y_pred, y_true, average='macro').values
    print(score)
    diff = sqrt(((true_scores - score) ** 2).sum())
    assert diff < 1e-5

def test_columnwise_confusion_matrix():

    '''
    first_column:
        0: Precision: 0.5, Recall: 1., F1: 2 / 3
        1: Precision: 1, Recall: 0,5, F1: 2 / 3
        macro: 2 / 3s
    second_column:
        0: F1: 1
        1: F1: 1
        macro: 1
    '''

    y_true = [
        [1, 0],
        [1, 1],
        [0, 1]
    ]

    y_pred = [
        [0, 0],
        [1, 1],
        [0, 1]
    ]

    categories = [0, 1,]
    matrices = columnwise_confusion_matrix(y_pred, y_true, categories)

    assert len(matrices) == 2

    matrix_0 = array([
        [1, 0],
        [1, 1],
    ])

    diff = sqrt(((matrix_0 - matrices[0].values) ** 2).sum())

    assert diff < 1e-5

from ..helpers.utils import _create_param_grid

def test__create_param_grid():
    initial_grid = {
        'first': [1, 2],
        "second": [2, 3]
    }

    final_grid = [
        {'first': 1, 'second': 2},
        {'first': 1, 'second': 3},
        {'first': 2, 'second': 2},
        {'first': 2, 'second': 3},
    ]

    def _serialize(list_of_dicts: list) -> str:

        list_of_str = sorted(list(map(str, list_of_dicts)))

        return str(list_of_str)

    final_grid = _serialize(final_grid)

    final_grid_res = _create_param_grid(initial_grid)
    final_grid_res = _serialize(final_grid_res)

    assert final_grid == final_grid_res

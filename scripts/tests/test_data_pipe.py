from turtle import shape
from pandas import read_csv

from scripts.pipeline.data_pipe import *
from scripts.pipeline.data_pipe import _select, _split_and_drop, _shuffle, _as_numpy
from scripts.pipeline.data_pipe import _reshape, _pack_with_array

from scripts.helpers.preprocess import preprocess_3h

DF = read_csv('scripts/tests/test_data/test.csv').pipe(preprocess_3h)

def test_simple_pipe():
    steps = [
        (_select, {"columns": ['Kp*10', 'doyCos', 'doySin']}, False)
    ]
    pipe = DataPipe(steps)
    result = pipe.fit_transform(DF)

    assert result.shape == (DF.shape[0], 3)

class TestDataBlocks:

    def test__select(self):
        df = _select(DF, ['Kp*10', 'doyCos', 'doySin'])
        assert str(df.columns.tolist()) == str(['Kp*10', 'doyCos', 'doySin'])
    
    def test__shuffle(self):
        df, cols = _shuffle((DF, []), shuffle=True, random_state=18)
        assert df.shape == DF.shape
    
    def test__split_and_drop(self):
        X, y = _split_and_drop((DF, ['Kp*10', 'doyCos', 'doySin']), drop_columns=['hourCos'])
        assert 'hourCos' not in X.columns
        assert 'doyCos' in y.columns
        assert X.shape[1] == DF.shape[1] - 4
    
    def test__as_numpy(self):
        df = _as_numpy((DF,))
        assert df[0].shape == DF.shape
    
    def test__reshape(self):
        data_tuple = _split_and_drop((DF, ['Kp*10', 'doyCos', 'doySin']), 
                                     drop_columns=['hourCos'])
        data_tuple =  data_tuple[0].iloc[:, :12].values, data_tuple[1]
        
        reshaped_data_tuple = _reshape(data_tuple, 3, 4)
        assert reshaped_data_tuple[0].shape[1:] == (4, 3)
    
    def test__pack_with_array(self):
        data_tuple = _split_and_drop((DF, ['Kp*10', 'doyCos', 'doySin']), 
                                     drop_columns=['hourCos'])
        data_tuple =  data_tuple[0].iloc[:, :12], data_tuple[1]
        
        packed = _pack_with_array(data_tuple, ['Kp*10', 'doyCos', 'doySin'])
        assert len(packed) == 3

from scripts.helpers.yaml_utils import load_yaml

CONF = load_yaml('scripts/tests/test_yamls/test_vars.yaml')

def test_lag_data_pipe():
    data_pipe = LagDataPipe(**CONF)
    X_train, y_train, features = data_pipe.fit_transform(DF)
    X_train, y_train, features = data_pipe.transform(DF)
    n_records = DF.shape[0] - CONF['forward_steps'] - CONF['backward_steps']
    assert y_train.shape == (n_records, CONF['forward_steps'])
    n_features = (CONF['backward_steps'] + 1) * len(CONF['variables'])
    assert X_train.shape == (n_records, n_features)
    assert len(features) == n_features

def test_sequence_data_pipe():
    data_pipe = SequenceDataPipe(**CONF)
    X_train, y_train, features = data_pipe.fit_transform(DF)
    X_train, y_train, features = data_pipe.transform(DF)
    n_records = DF.shape[0] - CONF['forward_steps'] - CONF['backward_steps']
    assert y_train.shape == (n_records, CONF['forward_steps'])
    n_features = len(CONF['variables'])
    time_steps = CONF['backward_steps'] + 1
    assert X_train.shape == (n_records, time_steps, n_features)
    assert len(features) == n_features

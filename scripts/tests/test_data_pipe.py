import os
from pandas import read_csv

from scripts.pipeline.data_pipe import *
from scripts.pipeline.preprocess import _select

from scripts.pipeline.preprocess import preprocess_3h

LOCATION = os.path.dirname(os.path.realpath(__file__))
TEST_FILE_LOCATION = os.path.join(LOCATION, "test_data/test.csv")

# def test_simple_pipe():

#     out_location  = os.path.join(LOCATION, "out")
#     pipe = DataPipe(TEST_FILE_LOCATION, out_location)
#     pipe.load()

#     assert len(pipe.categories.unique()) == 3
#     assert pipe.data.shape[0] > 1

#     assert os.path.exists(pipe.out_path_test)
#     assert os.path.exists(pipe.out_path_train)




# from scripts.helpers.yaml_utils import load_yaml

# CONF = load_yaml(os.path.join(LOCATION, "test_yamls/test_vars.yaml"))


# def test_lag_data_pipe():
#     data_pipe = LagDataPipe(**CONF)
#     X_train, y_train, features = data_pipe.fit_transform(DF)
#     X_train, y_train, features = data_pipe.transform(DF)
#     n_records = DF.shape[0] - CONF["forward_steps"] - CONF["backward_steps"]
#     assert y_train.shape == (n_records, CONF["forward_steps"])
#     n_features = (CONF["backward_steps"] + 1) * len(CONF["variables"])
#     assert X_train.shape == (n_records, n_features)
#     assert len(features) == n_features


# def test_sequence_data_pipe():
#     data_pipe = SequenceDataPipe(**CONF)
#     X_train, y_train, features = data_pipe.fit_transform(DF)
#     X_train, y_train, features = data_pipe.transform(DF)
#     n_records = DF.shape[0] - CONF["forward_steps"] - CONF["backward_steps"]
#     assert y_train.shape == (n_records, CONF["forward_steps"])
#     n_features = len(CONF["variables"])
#     time_steps = CONF["backward_steps"] + 1
#     assert X_train.shape == (n_records, time_steps, n_features)
#     assert len(features) == n_features

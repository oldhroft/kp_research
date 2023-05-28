import os

from pandas import read_csv

from scripts.pipeline.preprocess import *
from scripts.helpers.yaml_utils import load_yaml


LOCATION = os.path.dirname(os.path.realpath(__file__))


def test_categorize():
    assert categorize(40) == 1
    assert categorize(70) == 2
    assert categorize(30) == 0


DF = read_csv(os.path.join(LOCATION, "test_data/test.csv"))


def test_preprocess_3h():
    conf = load_yaml(os.path.join(LOCATION, "test_yamls/test_vars.yaml"))
    df = DF.pipe(preprocess_3h)[conf["variables"]]
    assert df.isna().sum().sum() == 0, "NA values are present"
    assert df.shape[0] == DF.shape[0] // 3 + 1, "Shape is not preserved"


from scripts.pipeline.preprocess import _select, _split_and_drop, _shuffle, _as_numpy
from scripts.pipeline.preprocess import _reshape, _pack_with_array


class TestDataBlocks:
    def test__select(self):
        df = _select(DF, ["Kp*10", "doyCos", "doySin"])
        assert str(df.columns.tolist()) == str(["Kp*10", "doyCos", "doySin"])

    def test__shuffle(self):
        df, cols = _shuffle((DF, []), shuffle=True, random_state=18)
        assert df.shape == DF.shape

    def test__split_and_drop(self):
        X, y = _split_and_drop(
            (DF, ["Kp*10", "doyCos", "doySin"]), drop_columns=["hourCos"]
        )
        assert "hourCos" not in X.columns
        assert "doyCos" in y.columns
        assert X.shape[1] == DF.shape[1] - 4

    def test__as_numpy(self):
        df = _as_numpy((DF,))
        assert df[0].shape == DF.shape

    def test__reshape(self):
        data_tuple = _split_and_drop(
            (DF, ["Kp*10", "doyCos", "doySin"]), drop_columns=["hourCos"]
        )
        data_tuple = data_tuple[0].iloc[:, :12].values, data_tuple[1]

        reshaped_data_tuple = _reshape(data_tuple, 3, 4)
        assert reshaped_data_tuple[0].shape[1:] == (4, 3)

    def test__pack_with_array(self):
        data_tuple = _split_and_drop(
            (DF, ["Kp*10", "doyCos", "doySin"]), drop_columns=["hourCos"]
        )
        data_tuple = data_tuple[0].iloc[:, :12], data_tuple[1]

        packed = _pack_with_array(data_tuple, ["Kp*10", "doyCos", "doySin"])
        assert len(packed) == 3

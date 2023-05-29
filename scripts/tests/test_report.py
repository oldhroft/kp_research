import os

from pandas import read_csv
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from numpy.random import choice

from scripts.report.generate_report import basename, get_feature_importances

LOCATION = os.path.dirname(os.path.realpath(__file__))

def test_basename():
    assert basename("/mypath/file.csv") == "file"


# def test_get_feature_importances():
#     data = (
#         read_csv(os.path.join(LOCATION, "test_data/test.csv"))
#         .select_dtypes(include=["int64", "float64"])
#         .fillna(0)
#     )
#     X = data.values[:, :-1]
#     y = data.values[:, -1]
#     columns = data.columns[:-1]

#     rg = Ridge()
#     rg.fit(X, y)
#     imp = get_feature_importances(rg, data.columns[:-1])
#     assert imp.shape[0] == len(columns)
#     assert imp.shape[1] == 1

#     rf = RandomForestRegressor()
#     rf.fit(X, y)
#     imp = get_feature_importances(rf, columns)
#     assert imp.shape[0] == len(columns)
#     assert imp.shape[1] == 1

#     pp = make_pipeline(StandardScaler(), Ridge())
#     pp.fit(X, y)
#     imp = get_feature_importances(pp, columns)
#     assert imp.shape[0] == len(columns)
#     assert imp.shape[1] == 1

#     pp = make_pipeline(StandardScaler(), RandomForestRegressor())
#     pp.fit(X, y)
#     imp = get_feature_importances(pp, columns)
#     assert imp.shape[0] == len(columns)
#     assert imp.shape[1] == 1

#     y = choice([0, 1], size=len(X))
#     pp = make_pipeline(StandardScaler(), RidgeClassifier())
#     pp.fit(X, y)
#     imp = get_feature_importances(pp, columns)
#     assert imp.shape[0] == len(columns)
#     assert imp.shape[1] == 1

#     y = choice([0, 1, 2], size=len(X))
#     pp = make_pipeline(StandardScaler(), RidgeClassifier())
#     pp.fit(X, y)
#     imp = get_feature_importances(pp, columns)
#     assert imp.shape[0] == len(columns)
#     assert imp.shape[1] == 3


# from sklearn.multioutput import MultiOutputClassifier
# from sklearn.ensemble import RandomForestClassifier
# from scripts.report.generate_report import get_feature_importances_multi_output


# def test_get_feature_importances_multi_output():

#     data = (
#         read_csv(os.path.join(LOCATION, "test_data/test.csv"))
#         .select_dtypes(include=["int64", "float64"])
#         .fillna(0)
#     )
#     X = data.values[:, :-1]
#     y = choice([0, 1, 2], size=(len(X), 2))
#     columns = data.columns[:-1]

#     pp = MultiOutputClassifier(make_pipeline(StandardScaler(), RidgeClassifier()))

#     pp.fit(X, y)
#     imp = get_feature_importances_multi_output(pp, columns)
#     assert imp.shape[0] == len(columns)
#     assert imp.shape[1] == 6

#     pp = MultiOutputClassifier(
#         make_pipeline(StandardScaler(), RandomForestClassifier())
#     )

#     pp.fit(X, y)
#     imp = get_feature_importances_multi_output(pp, columns)
#     assert imp.shape[0] == len(columns)
#     assert imp.shape[1] == 2


import os
import pytest
from pandas import read_excel
from scripts.report.generate_report import concat_save
from scripts.helpers.utils import create_folder


@pytest.fixture
def result_folder():
    fld = os.path.join(LOCATION, "test_concat_result")
    create_folder(fld)
    yield fld
    os.system(f"rm -rf {fld}")


# def test_concat_save(result_folder):
#     res_path = os.path.join(result_folder, 'result.xlsx')
    
#     concat_save(os.path.join(LOCATION, "test_concat"), res_path, "csv")
#     assert os.path.exists(res_path)

#     df = read_excel(res_path, index_col=0)
#     assert df.shape[0] == 8
#     assert df.shape[1] == 2


from scripts.report.generate_report import extract_fi_from_models
import joblib
import yaml
import glob


@pytest.fixture
def model_folder():
    fld = os.path.join(LOCATION, 'test_model_fi')
    create_folder(fld)
    yield fld
    os.system(f"rm -rf {fld}")


@pytest.fixture
def fi_folder():
    fld = os.path.join(LOCATION, 'fi_folder')
    create_folder(fld)
    yield fld
    os.system(f"rm - rf {fld}")


# def test_extract_fi_from_models(model_folder, fi_folder):

#     data = (
#         read_csv(os.path.join(LOCATION, "test_data/test.csv"))
#         .select_dtypes(include=["int64", "float64"])
#         .fillna(0)
#     )
#     X = data.values[:, :-1]
#     y = choice([0, 1, 2], size=(len(X), 2))
#     columns = list(data.columns[:-1])

#     pp = MultiOutputClassifier(make_pipeline(StandardScaler(), RidgeClassifier()))
#     pp.fit(X, y)

#     model_dump_folder = os.path.join(model_folder, "model")
#     create_folder(model_dump_folder)

#     vars_folder = os.path.join(model_folder, "vars")
#     create_folder(vars_folder)

#     model_vars = {"features": columns}

#     for i in range(3):
#         with open(
#             os.path.join(vars_folder, f"vars_model{i}.yaml"), "w", encoding="utf-8"
#         ) as file:
#             yaml.dump(model_vars, file)

#     for i in range(3):
#         joblib.dump(pp, os.path.join(model_dump_folder, f"model{i}_model.pkl"))

#     extract_fi_from_models(model_folder, fi_folder)

#     files = glob.glob(os.path.join(fi_folder, "*.xlsx"))
#     assert len(files) == 3

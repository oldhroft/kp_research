import glob
import os
from typing import Any, List
import joblib

from argparse import ArgumentParser

from pandas import concat, read_csv, DataFrame
from tqdm import tqdm

from scripts.helpers.utils import create_folder
from scripts.helpers.yaml_utils import load_yaml


def create_argparser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--folder", action="store", type=str, required=True)
    parser.add_argument("--imp", action="store_true")
    return parser


def basename(x: str) -> str:
    return os.path.basename(x).split(".")[0]


def get_feature_importances(model: Any, features: List) -> DataFrame:
    if hasattr(model, "coef_"):
        if len(model.coef_.shape) == 1:
            importances = [model.coef_]
        else:
            importances = list(model.coef_)
    elif hasattr(model, "feature_importances_"):
        importances = [model.feature_importances_]
    elif hasattr(model, "steps"):
        last_estimator = model.steps[-1][1]
        if hasattr(last_estimator, "coef_"):
            if len(last_estimator.coef_.shape) == 1:
                importances = [last_estimator.coef_]
            else:
                importances = list(last_estimator.coef_)
        elif hasattr(last_estimator, "feature_importances_"):
            importances = [last_estimator.feature_importances_]
        else:
            importances = []
    else:
        importances = []

    return DataFrame(importances, columns=features).T


def get_feature_importances_multi_output(model: Any, features: list) -> DataFrame:
    importances = []
    for estimator in model.estimators_:
        importances.append(get_feature_importances(estimator, features))
    return concat(importances, axis=1, ignore_index=True)


def concat_save(folder: str, output_path: str, format: str = "csv") -> None:
    def read_file(fname: str) -> DataFrame:
        name = basename(fname)
        return read_csv(fname, index_col=0,).iloc[:, 0].rename(name).to_frame()

    format = f"*.{format}"
    files = glob.glob(os.path.join(folder, format))
    df = concat(
        (read_file(file) for file in files),
        axis=1,
    )
    df.to_excel(output_path)


def extract_fi_from_models(folder: str, importances_folder: str) -> None:
    models_wildcard = os.path.join(os.path.join(folder, "model"), "*.pkl")
    for model_path in tqdm(glob.glob(models_wildcard)):
        filename = basename(model_path)
        model_name = "_".join(filename.split("_")[:-1])
        model = joblib.load(model_path)
        vars_path = os.path.join(folder, "vars")
        features = load_yaml(os.path.join(vars_path, f"vars_{model_name}.yaml"))[
            "features"
        ]
        importances = get_feature_importances_multi_output(model, features)
        if len(importances) > 0:
            importances.to_excel(
                os.path.join(
                    importances_folder, f"feature_importances_{model_name}.xlsx"
                )
            )

def generate_report():
    arguments = create_argparser().parse_args()
    folder = arguments.folder
    report_path = os.path.join(folder, "report")
    create_folder(report_path)
    concat_save(folder, os.path.join(report_path, "report.xlsx"), "csv")

    if arguments.imp:
        importances_folder = os.path.join(report_path, "feature_importances")
        create_folder(importances_folder)
        extract_fi_from_models(folder, importances_folder)

if __name__ == "__main__":
    generate_report()
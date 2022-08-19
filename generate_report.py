import glob
import os
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


def read_file(fname: str) -> DataFrame:
    name = basename(fname)
    return read_csv(fname, index_col=0, squeeze=True).rename(name).to_frame()


def _extract_feature_importances(model, features):

    importances = {}
    k = 0
    for estimator in model.estimators_:
        if hasattr(estimator, "coef_"):
            for j in range(len(estimator.coef_)):
                importances[k] = estimator.coef_[j]
                k += 1
        elif hasattr(estimator, "feature_importances_"):
            importances[k] = estimator.feature_importances_
            k += 1
        elif hasattr(estimator, "steps"):
            last_estimator = estimator.steps[-1][1]
            if hasattr(last_estimator, "coef_"):
                for j in range(len(last_estimator.coef_)):
                    imp = last_estimator.coef_[j]
                    if len(imp) == len(features):
                        importances[k] = imp
                        k += 1
            elif hasattr(last_estimator, "feature_importances_"):
                imp = last_estimator.feature_importances_
                if len(imp) == len(features):
                    importances[k] = imp
                    k += 1

        else:
            continue

    return DataFrame(importances, index=features)


if __name__ == "__main__":

    arguments = create_argparser().parse_args()
    folder = arguments.folder
    report_path = os.path.join(folder, "report")
    create_folder(report_path)
    files = glob.glob(os.path.join(folder, "*.csv"))

    df = concat(
        (read_file(file) for file in files),
        axis=1,
    )
    df.to_excel(os.path.join(report_path, "report.xlsx"))

    if arguments.imp:
        importances_folder = os.path.join(report_path, "feature_importances")
        create_folder(importances_folder)
        models_wildcard = os.path.join(os.path.join(folder, "model"), "*.pkl")
        for model_path in tqdm(glob.glob(models_wildcard)):
            filename = basename(model_path)
            model_name = "_".join(filename.split("_")[:-1])
            model = joblib.load(model_path)
            vars_path = os.path.join(folder, "vars")
            features = load_yaml(os.path.join(vars_path, f"vars_{model_name}.yaml"))[
                "features"
            ]
            importances = _extract_feature_importances(model, features)
            if len(importances) > 0:
                importances.to_excel(
                    os.path.join(
                        importances_folder, f"feature_importances_{model_name}.xlsx"
                    )
                )

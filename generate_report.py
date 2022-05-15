import glob
import os
import joblib

from argparse import ArgumentParser

from pandas import concat, read_csv, DataFrame, Series

from scripts.helpers.utils import create_folder

def create_argparser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--folder', action='store', type=str, required=True)
    parser.add_argument('--imp', action='store_true')
    return parser

def basename(x: str) -> str:
    return os.path.basename(x).split('.')[0]

def read_file(fname: str) -> DataFrame:
    name = basename(fname)
    return (
        read_csv(fname, index_col=0, squeeze=True)
        .rename(name)
        .to_frame())

def _extract_feature_importances(model):

    importances = {}

    for i, estimator in enumerate(model.estimators_):
        if hasattr(estimator, 'coef_'):
            importances[i] = estimator.coef_
        elif hasattr(estimator, 'feature_importances_'):
            importances[i] = estimator.feature_importances_
        else:
            continue
    
    return DataFrame(importances)

if __name__ == '__main__':

    arguments = create_argparser().parse_args()
    folder = arguments.folder
    report_path = os.path.join(folder, 'report')
    create_folder(report_path)
    files = glob.glob(os.path.join(folder, '*.csv'))

    df = concat((read_file(file) for file in files),
                axis=1, )
    df.to_excel(os.path.join(report_path, 'report.xlsx'))

    if arguments.imp:
        importances_folder = os.path.join(report_path, 'feature_importances')
        create_folder(importances_folder)
        models_wildcard = os.path.join(os.path.join(folder, 'model'), '*.pkl')
        for model_path in glob.glob(models_wildcard):
            filename = basename(model_path)
            model_name = '_'.join(filename.split('_')[: -1])
            model = joblib.load(model_path)
            importances = _extract_feature_importances(model)
            if len(importances) > 0:
                importances.to_excel(os.path.join(importances_folder, 
                                                  f'feature_importances_{model_name}.xlsx'))




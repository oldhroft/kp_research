import glob
import os

from argparse import ArgumentParser

from pandas import concat, read_csv, DataFrame

def create_argparser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--folder', action='store', type=str, required=True)
    return parser

def basename(x: str) -> str:
    return os.path.basename(x).split('.')[0]

def read_file(fname: str) -> DataFrame:
    name = basename(fname)
    return (
        read_csv(fname, index_col=0, squeeze=True)
        .rename(name)
        .to_frame())

if __name__ == '__main__':

    arguments = create_argparser().parse_args()
    folder = arguments.folder
    files = glob.glob(os.path.join(folder, '*.csv'))

    df = concat((read_file(file) for file in files),
                axis=1, )
    df.to_excel(os.path.join(folder, 'report.xlsx'))



from typing import Any, List, Tuple, Union
import os

from pandas import read_csv, DataFrame
from numpy import ndarray
import datetime

from scripts.helpers.utils import add_lags
from scripts.pipeline.preprocess import categorize


def read_data(path):
    df = read_csv(path, encoding="cp1251", na_values="N")
    return df


def split_data(data: DataFrame, year_test: int = 2020) -> Tuple[DataFrame]:
    return (
        data.loc[data.year >= year_test].reset_index(drop=True),
        data.loc[data.year < year_test].reset_index(drop=True),
    )


DEFAULT_BORDERS = [
    17,
    33,
]


class DataPipe:
    def __init__(
        self,
        path: str,
        out_folder: str,
        date_from: str = None,
        year_test: int = 2020,
        categories: List[int] = DEFAULT_BORDERS,
    ) -> None:
        self.path = path
        self.out_folder = out_folder
        self.date_from = date_from
        self.features = []
        self.targets = []
        self.year_test = year_test
        self.categories = categories

    def extract(
        self,
    ) -> None:
        self.data_raw = read_data(self.path)

    def transform(self) -> None:
        if self.date_from is not None:
            self.data = self.data_raw.loc[
                self.data_raw.dttm >= self.date_from
            ].reset_index(drop=True)
        else:
            self.data = self.data_raw.copy()
        self.data["dttm"] = self.data.apply(
            lambda y: datetime.datetime(
                int(y.year), int(y.month), int(y.day), int(y["hour from"]), 0
            ),
            axis=1,
        )

        self.data["category"] = self.data["Kp*10"].apply(
            categorize, categories=self.categories
        )

    def load(self) -> None:
        self.extract()
        self.transform()

        self.data_train, self.data_test = split_data(self.data, self.year_test)

        self.out_path_train = os.path.join(
            self.out_folder, f"{self.__class__.__name__}_train.parquet"
        )
        self.out_path_test = os.path.join(
            self.out_folder, f"{self.__class__.__name__}_test.parquet"
        )

        self.data_train.to_parquet(self.out_path_train)
        self.data_test.to_parquet(self.out_path_test)


class DataPipe3H(DataPipe):
    def __init__(
        self,
        path: str,
        out_folder: str,
        date_from: str = None,
        year_test: int = 2020,
        categories: List[int] = DEFAULT_BORDERS,
    ) -> None:
        super().__init__(path, out_folder, date_from, year_test, categories)

    def transform(self) -> None:
        super().transform()
        self.data = self.data.sort_values(by="dttm").iloc[::3].bfill()


class LagDataPipe3H(DataPipe3H):
    def __init__(
        self,
        path: str,
        out_folder: str,
        variables: list,
        target: Any,
        backward_steps: int,
        forward_steps: int,
        scale: bool = False,
        shuffle: bool = True,
        random_state: Union[int, None] = None,
        date_from: str = None,
        year_test: int = 2020,
        categories: List[int] = DEFAULT_BORDERS,
    ) -> None:
        super().__init__(path, out_folder, date_from, year_test, categories)

        self.variables = variables
        self.target = target
        self.backward_steps = backward_steps
        self.forward_steps = forward_steps
        self.scale = scale
        self.shuffle = shuffle
        self.random_state = random_state

    def transform(self) -> None:
        super().transform()

        self.data, features = add_lags(
            self.data,
            lags=self.backward_steps,
            forward=False,
            trim=True,
            subset=self.variables,
            return_cols=True,
        )

        self.data, targets = add_lags(
            self.data,
            lags=self.forward_steps,
            forward=False,
            trim=True,
            subset=self.target,
            return_cols=True,
        )

        self.features = features
        self.targets = targets

    def get_xy(self) -> Tuple[ndarray]:
        if self.scale:
            mean = self.data[self.features].mean()
            std = self.data[self.features].std()

            return (
                ((self.data_train[self.features] - mean) / std).values,
                self.data_train[self.targets].values,
                ((self.data_test[self.features] - mean) / std).values,
                self.data_test[self.targets].values,
            )
        else:
            return (
                self.data_train[self.features].values,
                self.data_train[self.targets].values,
                self.data_test[self.features].values,
                self.data_test[self.targets].values,
            )


class SequenceDataPipe3H(LagDataPipe3H):
    def __init__(
        self,
        path: str,
        out_folder: str,
        variables: list,
        target: Any,
        backward_steps: int,
        forward_steps: int,
        scale: bool = False,
        shuffle: bool = True,
        random_state: int | None = None,
        date_from: str = None,
        year_test: int = 2020,
        categories: List[int] = DEFAULT_BORDERS,
    ) -> None:
        super().__init__(
            path,
            out_folder,
            variables,
            target,
            backward_steps,
            forward_steps,
            scale,
            shuffle,
            random_state,
            date_from,
            year_test,
            categories,
        )

    def get_xy(self) -> Tuple[ndarray]:
        X_train, y_train, X_test, y_test = super().get_xy()

        return (
            X_train.reshape(-1, self.forward_steps, len(self.features)),
            y_train,
            X_test.reshape(-1, self.forward_steps, len(self.features)),
            y_test,
        )


# class DataPipe(object):
#     def __init__(self, steps: List[Tuple[Any, Any, bool]]) -> None:
#         self.steps = steps

#     def fit_transform(self, X: Any):

#         self.fitted_steps = []
#         for step, arg, is_fittable in self.steps:

#             if is_fittable:
#                 X = step.fit_transform(X, **arg)
#                 step = step.transform
#             else:
#                 X = step(X, **arg)
#             self.fitted_steps.append((step, arg))
#         return X

#     def transform(self, X: Any):
#         for step, arg in self.fitted_steps:
#             X = step(X, **arg)
#         return X


# from scripts.helpers.utils import add_lags
# from scripts.pipeline.preprocess import (
#     _as_numpy,
#     _get_feature_names,
#     _select,
#     _shuffle,
#     _split_and_drop,
# )

# from scripts.pipeline.preprocess import _StandardScalerXY


# class LagDataPipe(DataPipe):
#     def __init__(
#         self,
#         variables: list,
#         target: Any,
#         backward_steps: int,
#         forward_steps: int,
#         scale: bool = False,
#         shuffle: bool = True,
#         random_state: Union[int, None] = None,
#     ) -> None:
#         self.steps = [
#             (_select, {"columns": variables + [target]}, False),
#             (
#                 add_lags,
#                 {
#                     "lags": backward_steps,
#                     "forward": False,
#                     "trim": True,
#                     "subset": variables,
#                     "return_cols": False,
#                 },
#                 False,
#             ),
#             (
#                 add_lags,
#                 {
#                     "lags": forward_steps,
#                     "forward": True,
#                     "trim": True,
#                     "subset": target,
#                     "return_cols": True,
#                 },
#                 False,
#             ),
#             (_shuffle, {"random_state": random_state, "shuffle": shuffle}, False),
#             (_split_and_drop, {"drop_columns": [target]}, False),
#             (_get_feature_names, {}, False),
#             (_as_numpy, {}, False),
#         ]
#         if scale:
#             self.steps.append((_StandardScalerXY(), {}, True))


# from scripts.pipeline.preprocess import _reshape, _pack_with_array


# class SequenceDataPipe(LagDataPipe):
#     def __init__(
#         self,
#         variables: list,
#         target: Any,
#         backward_steps: int,
#         forward_steps: int,
#         scale: bool = False,
#         shuffle: bool = True,
#         random_state: Union[int, None] = None,
#     ):
#         super().__init__(
#             variables,
#             target,
#             backward_steps,
#             forward_steps,
#             scale=scale,
#             shuffle=shuffle,
#             random_state=random_state,
#         )

#         self.steps.extend(
#             [
#                 (
#                     _reshape,
#                     {"n_features": len(variables), "time_steps": backward_steps + 1},
#                     False,
#                 ),
#                 (_pack_with_array, {"array": variables}, False),
#             ]
#         )

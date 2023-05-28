from typing import Any, List, Tuple, Union


class DataPipe(object):
    def __init__(self, steps: List[Tuple[Any, Any, bool]]) -> None:
        self.steps = steps

    def fit_transform(self, X: Any):

        self.fitted_steps = []
        for step, arg, is_fittable in self.steps:

            if is_fittable:
                X = step.fit_transform(X, **arg)
                step = step.transform
            else:
                X = step(X, **arg)
            self.fitted_steps.append((step, arg))
        return X

    def transform(self, X: Any):
        for step, arg in self.fitted_steps:
            X = step(X, **arg)
        return X


from scripts.helpers.utils import add_lags
from scripts.pipeline.preprocess import (
    _as_numpy,
    _get_feature_names,
    _select,
    _shuffle,
    _split_and_drop,
)
from scripts.pipeline.preprocess import _StandardScalerXY


class LagDataPipe(DataPipe):
    def __init__(
        self,
        variables: list,
        target: Any,
        backward_steps: int,
        forward_steps: int,
        scale: bool = False,
        shuffle: bool = True,
        random_state: Union[int, None] = None,
    ) -> None:
        self.steps = [
            (_select, {"columns": variables + [target]}, False),
            (
                add_lags,
                {
                    "lags": backward_steps,
                    "forward": False,
                    "trim": True,
                    "subset": variables,
                    "return_cols": False,
                },
                False,
            ),
            (
                add_lags,
                {
                    "lags": forward_steps,
                    "forward": True,
                    "trim": True,
                    "subset": target,
                    "return_cols": True,
                },
                False,
            ),
            (_shuffle, {"random_state": random_state, "shuffle": shuffle}, False),
            (_split_and_drop, {"drop_columns": [target]}, False),
            (_get_feature_names, {}, False),
            (_as_numpy, {}, False),
        ]
        if scale:
            self.steps.append((_StandardScalerXY(), {}, True))


from scripts.pipeline.preprocess import _reshape, _pack_with_array


class SequenceDataPipe(LagDataPipe):
    def __init__(
        self,
        variables: list,
        target: Any,
        backward_steps: int,
        forward_steps: int,
        scale: bool = False,
        shuffle: bool = True,
        random_state: Union[int, None] = None,
    ):

        super().__init__(
            variables,
            target,
            backward_steps,
            forward_steps,
            scale=scale,
            shuffle=shuffle,
            random_state=random_state,
        )

        self.steps.extend(
            [
                (
                    _reshape,
                    {"n_features": len(variables), "time_steps": backward_steps + 1},
                    False,
                ),
                (_pack_with_array, {"array": variables}, False),
            ]
        )

from pandas import DataFrame

from scripts.pipeline.data_pipe import *
from scripts.pipeline.data_pipe import _select

df = [
    ('first', 'second', 'third', 'fourth', 'fifth')
]
df = DataFrame(df, columns=['first', 'second', 'third', 'fourth', 'fifth'])


def test_simple_pipe():
    steps = [
        (_select, {"columns": ['first', 'second', 'third']}, False)
    ]
    pipe = DataPipe(steps)
    result = pipe.fit_transform(df)

    assert result.shape == (1, 3)
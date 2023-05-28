import datetime

def categorize(x):
    if x <= 17: return 0
    elif x <= 33: return 1
    else: return 2

def preprocess_3h(df, regression=False):

    x = df.copy()
    if not regression:
        x['category'] = x['Kp*10'].apply(categorize)
    else:
        x["category"] = x["Kp*10"]

    x["dttm"] = x.apply(
        lambda y: datetime.datetime(
            int(y.year), int(y.month), int(y.day), int(y["hour from"]), 0
        ),
        axis=1,
    )

    x_3h = x.sort_values(by="dttm").iloc[::3].bfill()
    return x_3h


from sklearn.preprocessing import StandardScaler
from numpy import array


def _select(df: DataFrame, columns: list) -> DataFrame:
    return df.loc[:, columns]


def _split_and_drop(data_tuple: tuple, drop_columns: list) -> tuple:
    df = data_tuple[0]
    y_columns = data_tuple[1]
    return df.drop(y_columns + drop_columns, axis=1), df.loc[:, y_columns]


def _get_feature_names(
    data_tuple: tuple,
) -> tuple:
    return data_tuple[0], data_tuple[1], data_tuple[0].columns


def _shuffle(data_tuple: tuple, shuffle: bool, random_state: int) -> DataFrame:
    if shuffle:
        X = data_tuple[0]
        cols = data_tuple[1]
        return X.sample(frac=1.0, random_state=random_state), cols
    else:
        return data_tuple


def _as_numpy(data_tuple: tuple) -> tuple:
    return tuple(map(array, data_tuple))


class _StandardScalerXY(StandardScaler):
    def fit(self, data_tuple: tuple) -> None:
        return super().fit(data_tuple[0])

    def transform(
        self,
        data_tuple: tuple,
    ) -> tuple:
        X_scaled = super().transform(data_tuple[0])
        return tuple([X_scaled, *data_tuple[1:]])

    def fit_transform(self, data_tuple: tuple) -> tuple:
        return self.fit(data_tuple).transform(data_tuple)


def _reshape(data_tuple, n_features, time_steps) -> tuple:
    X = data_tuple[0].reshape((-1, time_steps, n_features))
    y = data_tuple[1]
    return X, y


def _pack_with_array(data_tuple, array) -> tuple:
    return tuple((*data_tuple, array))

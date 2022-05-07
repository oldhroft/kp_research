from tensorflow.keras.models import Sequential
from tensorflow.keras import layers as L
from tensorflow.keras import callbacks as callbacks


def get_sequential_model(input_shape: tuple, n_classes: int=3, units_array: list=[10], 
                         optimizer: str='adam') -> Sequential:

    model = Sequential([
        L.Input(shape=input_shape),
        *(L.Dense(units=units, activation='relu') for units in units_array),
        L.Dense(units=n_classes, activation='softmax')  
    ])

    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    return model

        
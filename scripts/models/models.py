from tensorflow.keras.models import Sequential
from tensorflow.keras import layers as L


def get_sequential_model(input_shape: tuple, n_classes: int=3, units_array: list=[10], 
                         optimizer: str='adam') -> Sequential:

    model = Sequential([
        L.Input(shape=input_shape),
        *(L.Dense(units=units, activation='relu') for units in units_array),
        L.Dense(units=n_classes, activation='softmax')  
    ])

    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    return model

def get_lstm_model(input_shape, n_classes, units_array, optimizer, ):
    
    model = Sequential([
        L.Input(shape=input_shape),
        *(L.LSTM(i, return_sequences=True, ) 
          for i in units_array['rnn'][:-1]),
        L.LSTM(units_array['rnn'][-1]),
        *(L.Dense(units=units, activation='relu') 
          for units in units_array['dense']),
        L.Dense(n_classes, activation='softmax')
    ], )
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model

def get_gru_model(input_shape, n_classes, units_array, optimizer, ):
    
    model = Sequential([
        L.Input(shape=input_shape),
        *(L.GRU(i, return_sequences=True, ) 
          for i in units_array['rnn'][:-1]),
        L.GRU(units_array['rnn'][-1]),
        *(L.Dense(units=units, activation='relu') 
          for units in units_array['dense']),
        L.Dense(n_classes, activation='softmax')
    ], )
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model

        
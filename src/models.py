from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, Dropout
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping


def create_lstm_model(num_window, num_features, num_hidden=2, num_units=64, latent_unit=8, activation='tanh'):
    """
    커스터마이징이 가능한 LSTM 오토인코더 모델 생성 함수

    :param num_window: input data의 윈도우 크기
    :param num_features: input data의 feature 개수
    :param num_hidden: input layer~latent layer 사이의 LSTM layer 수
    :param num_units: 첫 LSTM layer의 unit 수
    :param latent_unit: latent vector 크기
    :param activation: LSTM layer의 activation function
    :return:
    """
    if latent_unit > num_units // (num_hidden * 2):
        raise AssertionError('LSTM unit 수가 latent vector 크기보다 작습니다.')

    init_unit = num_units
    model = Sequential()
    model.add(Input(shape=(num_window, num_features)))
    model.add(LSTM(init_unit, activation=activation, return_sequences=True))
    model.add(Dropout(0.2))
    for i in range(num_hidden - 1):
        num_units //= 2
        model.add(LSTM(num_units, activation=activation, return_sequences=True))
    model.add(LSTM(latent_unit, activation=activation, return_sequences=False))
    model.add(RepeatVector(num_window))
    for i in range(num_hidden - 1):
        model.add(LSTM(num_units, activation=activation, return_sequences=True))
        num_units *= 2
    model.add(LSTM(init_unit, activation=activation, return_sequences=True))
    model.add(TimeDistributed(Dense(num_features)))
    return model


def create_ocnn_model(train_data, num_window, num_features):
    # def train_auto_encoder_model():
    #     model = None
    #     return model
    ae_model = create_lstm_model(num_window, num_features)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    ae_model.fit(train_data, train_data, batch_size=32, epochs=3, validation_split=0.2,
                 callbacks=[early_stopping])
    encoder_model = Model(ae_model.layers[0].input, ae_model.layers[3].output)
    hidden_layer = Dense(100)(encoder_model.output)
    output_layer = Dense(1)(hidden_layer)
    model = Model(encoder_model.layers[0].input, output_layer)
    return model


def create_deepant_model(num_window, num_features):
    model = Sequential()
    model.add(Input(shape=(num_window, num_features)))
    model.add(Conv1D(filters=32, kernel_size=2, strides=1, padding='valid',
                     activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=32, kernel_size=2, strides=1, padding='valid',
                     activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(units=1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=num_window))
    return model

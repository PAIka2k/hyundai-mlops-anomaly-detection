# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pickle
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import datetime

signals = ['cell_01', 'cell_02', 'cell_03', 'cell_04', 'cell_05',
           'cell_06', 'cell_07', 'cell_08', 'cell_09', 'cell_10', 'cell_11',
           'cell_12', 'cell_13', 'cell_14', 'cell_15', 'cell_16', 'cell_17',
           'cell_18', 'cell_19', 'cell_20', 'cell_21', 'cell_22', 'cell_23',
           'cell_24', 'cell_25', 'cell_26', 'cell_27', 'cell_28', 'cell_29',
           'cell_30', 'cell_31', 'cell_32', 'cell_33', 'cell_34', 'cell_35',
           'cell_36', 'cell_37', 'cell_38', 'cell_39', 'cell_40', 'cell_41',
           'cell_42', 'cell_43', 'cell_44', 'cell_45', 'cell_46', 'cell_47',
           'cell_48', 'cell_49', 'cell_50', 'cell_51', 'cell_52', 'cell_53', 'cell_54', 'cell_55',
           'cell_56', 'cell_57', 'cell_58', 'cell_59', 'cell_60', 'cell_61',
           'cell_62', 'cell_63', 'cell_64', 'cell_65', 'cell_66', 'cell_67',
           'cell_68', 'cell_69', 'cell_70', 'cell_71', 'cell_72', 'cell_73',
           'cell_74', 'cell_75', 'cell_76', 'cell_77', 'cell_78', 'cell_79',
           'cell_80', 'cell_81', 'cell_82', 'cell_83', 'cell_84', 'cell_85',
           'cell_86', 'cell_87', 'cell_88', 'cell_89', 'cell_90', 'msr_data.ibm',
           'msr_data.r_isol', 'msr_data.vb_max', 'msr_data.vb_min',
           'msr_tbmax_raw', 'msr_tbmin_raw', 'SOC', 'CF_OBC_DCChargingStat', 'chg_charging_now', 'dnt', 'drv_cyc']

# %% 데이터 로딩
work_dir = './'
work_dir = 'D:/001 추진업무/002 ML기반 데이터분석 자동화도구개발_기술용역_2307~/회의/231005 데이터 및 코드 전달'
fname = 'pu_batt_sample10.csv'
data = pd.read_csv(os.path.join(work_dir, fname), header=0, usecols=signals)
data = data[signals]

data = data[(data['CF_OBC_DCChargingStat'] == 1) & (data['chg_charging_now'] == 1)]
data['delta_voltage'] = data.apply(lambda x: x['msr_data.vb_max'] - x['msr_data.vb_min'], axis=1)
data['delta_temp'] = data.apply(lambda x: x['msr_tbmax_raw'] - x['msr_tbmin_raw'], axis=1)

data.drop(['CF_OBC_DCChargingStat', 'chg_charging_now'], axis=1, inplace=True)

# %% 충전 번호(cycle_num) 생성
data.reset_index(drop=True, inplace=True)

# 날짜 형식 변환
data['dnt'] = data['dnt'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f'))
# 시간 차이 계산
data['time_diff'] = data['dnt'] - data['dnt'].shift(1)
# 시간 차이가 0.1초 초과하는 구간 index
cut_index = list(data[data.time_diff != datetime.timedelta(seconds=0.1)].index)
data.drop(['drv_cyc', 'dnt', 'time_diff'], axis=1, inplace=True)

cut_index += [len(data)]

charge_num_list = []
for charge_num in range(len(cut_index) - 1):
    count_num = cut_index[charge_num + 1] - cut_index[charge_num]
    charge_num_list += [charge_num] * count_num
data['cycle_num'] = charge_num_list

# scaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
scaled_data = pd.DataFrame(scaled_data, columns=data.columns)

# 전셀전압
vol_signals = [signal for signal in signals if signal.startswith('cell_')]
scaled_vol_data = scaled_data[vol_signals]

# 저항, 전류, 온도 등 전셀전압이외 데이터
scaled_other_data = scaled_data.drop(columns=vol_signals)

# %% 모델학습
# AE 전셀전압 차원축소
inputs = Input(shape=(scaled_vol_data.shape[1],))
encoded = Dense(64, activation='tanh')(inputs)
encoded = Dense(32, activation='tanh')(encoded)
encoded = Dense(10, activation='tanh')(encoded)
decoded = Dense(32, activation='tanh')(encoded)
decoded = Dense(64, activation='tanh')(decoded)
decoded = Dense(scaled_vol_data.shape[1])(decoded)
autoencoder = Model(inputs, decoded)
autoencoder.summary()

# %% Early stopping 콜백 정의
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

# AE 모델 컴파일 및 학습
optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
autoencoder.compile(optimizer=optimizer, loss='mse')

with tf.device('/device:GPU:0'):
    history = autoencoder.fit(scaled_vol_data, scaled_vol_data, batch_size=32, epochs=3, validation_split=0.2,
                              callbacks=[early_stopping])  # Early stopping 콜백 추가

# %% 인코더 모델 구성 (입력과 인코더 부분 연결)
encoder_model = Model(inputs=inputs, outputs=encoded)
encoder_model.summary()

# 차원축소된 전셀전압
with tf.device('/device:GPU:0'):
    # 입력 데이터에 대한 latent vector 추출
    latent_vector = encoder_model.predict(scaled_vol_data)

# %%
df_latent_vector = pd.DataFrame(latent_vector)

# LSTM_AE 입력 데이터
ad_train_data = pd.concat([scaled_other_data, df_latent_vector], axis=1)

# %% 입력데이터 윈도윙
window_size = 200
stride = 100
windowed_data = []

for i in range(int(max(ad_train_data['cycle_num'])) + 1):
    window1 = ad_train_data.loc[ad_train_data['cycle_num'] == i]

    for j in range(0, len(window1) - window_size, stride):
        window2 = window1.iloc[j:j + window_size]
        windowed_data.append(window2)

np_windows = np.array(windowed_data)
# cycle_num 컬럼 삭제
train_data = np.delete(np_windows, [9], axis=2)

# %% 이상탐지 모델 설계
inputs = Input(shape=(train_data.shape[1], train_data.shape[-1]))
encoded = LSTM(64, activation='tanh', return_sequences=True)(inputs)
encoded = Dropout(0.2)(encoded)
encoded = LSTM(32, activation='tanh', return_sequences=True)(encoded)
encoded = LSTM(8, activation='tanh', return_sequences=False)(encoded)
decoded = RepeatVector(train_data.shape[1])(encoded)
decoded = LSTM(32, activation='tanh', return_sequences=True)(decoded)
decoded = LSTM(64, activation='tanh', return_sequences=True)(decoded)
decoded = TimeDistributed(Dense(train_data.shape[-1]))(decoded)
autoencoder = Model(inputs, decoded)
autoencoder.summary()

# %% Early stopping 콜백 정의
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
autoencoder.compile(optimizer=optimizer, loss='mse')

with tf.device('/device:GPU:0'):
    history = autoencoder.fit(train_data, train_data, batch_size=32, epochs=3, validation_split=0.2,
                              callbacks=[early_stopping])  # Early stopping 콜백 추가

# %% Reconstruction error를 그래프로 그리기
plt.plot(history.history['loss'], c='b')
plt.plot(history.history['val_loss'], c='r')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend(['loss', 'val_loss'], loc='upper right')
plt.show()

# %% 모델 저장
scaler_fname = 'LSTM_AE_MinmaxScaler.pkl'
with open(os.path.join(work_dir, scaler_fname), 'wb') as f:
    pickle.dump(scaler, f)

encoder_fname = 'LSTM_AE_Dimension_Reduction_Model.h5'
encoder_model.save(os.path.join(work_dir, encoder_fname))

AE_fname = 'LSTM_AE_Model.h5'
autoencoder.save(os.path.join(work_dir, AE_fname))

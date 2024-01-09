#%% 파이썬 패키지 로딩
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.svm import OneClassSVM
import pickle

signals = ['cell_01', 'cell_02', 'cell_03', 'cell_04', 'cell_05',
       'cell_06', 'cell_07', 'cell_08', 'cell_09', 'cell_10', 'cell_11',
       'cell_12', 'cell_13', 'cell_14', 'cell_15', 'cell_16', 'cell_17',
       'cell_18', 'cell_19', 'cell_20', 'cell_21', 'cell_22', 'cell_23',
       'cell_24', 'cell_25', 'cell_26', 'cell_27', 'cell_28', 'cell_29',
       'cell_30', 'cell_31', 'cell_32', 'cell_33', 'cell_34', 'cell_35',
       'cell_36', 'cell_37', 'cell_38', 'cell_39', 'cell_40', 'cell_41',
       'cell_42', 'cell_43', 'cell_44', 'cell_45', 'cell_46', 'cell_47',
       'cell_48', 'cell_49','cell_50', 'cell_51', 'cell_52', 'cell_53', 'cell_54', 'cell_55',
       'cell_56', 'cell_57', 'cell_58', 'cell_59', 'cell_60', 'cell_61',
       'cell_62', 'cell_63', 'cell_64', 'cell_65', 'cell_66', 'cell_67',
       'cell_68', 'cell_69', 'cell_70', 'cell_71', 'cell_72', 'cell_73',
       'cell_74', 'cell_75', 'cell_76', 'cell_77', 'cell_78', 'cell_79',
       'cell_80', 'cell_81', 'cell_82', 'cell_83', 'cell_84', 'cell_85',
       'cell_86', 'cell_87', 'cell_88', 'cell_89', 'cell_90', 'msr_data.ibm',
       'msr_data.r_isol', 'msr_data.vb_max', 'msr_data.vb_min',
       'msr_tbmax_raw', 'msr_tbmin_raw', 'SOC', 'CF_OBC_DCChargingStat','chg_charging_now']

#%% 데이터 로딩
work_dir = './'
fname = 'pu_batt_sample10.csv'
data = pd.read_csv(os.path.join(work_dir, fname), header=0, usecols = signals)
data = data[signals]

data=data[(data['CF_OBC_DCChargingStat']==1)&(data['chg_charging_now']==1)]
data['delta_voltage'] = data.apply(lambda x : x['msr_data.vb_max'] - x['msr_data.vb_min'], axis=1)
data['delta_temp'] = data.apply(lambda x : x['msr_tbmax_raw'] - x['msr_tbmin_raw'], axis=1)

data.drop(['CF_OBC_DCChargingStat','chg_charging_now'],axis=1,inplace=True)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
scaled_data = pd.DataFrame(scaled_data, columns=data.columns)


# 전셀전압
vol_signals = [signal for signal in signals if signal.startswith('cell_')]
scaled_vol_data = scaled_data[vol_signals]


# 저항, 전류, 온도 등 전셀전압이외 데이터
scaled_other_data = scaled_data.drop(columns=vol_signals)

#%% 모델학습
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
    history = autoencoder.fit(scaled_vol_data, scaled_vol_data, batch_size=32, epochs=3, validation_split=0.2, callbacks=[early_stopping])  # Early stopping 콜백 추가

# %% 인코더 모델 구성 (입력과 인코더 부분 연결)
encoder_model = Model(inputs=inputs, outputs=encoded)
encoder_model.summary()

# 차원축소된 전셀전압
with tf.device('/device:GPU:0'):
    # 입력 데이터에 대한 latent vector 추출
    latent_vector = encoder_model.predict(scaled_vol_data)


# OCSVM 입력 데이터
ad_train_data = np.concatenate((scaled_other_data, latent_vector), axis=1)

# OCSVM 학습
anomal_ratio = 0.0005
ocsvm_model = OneClassSVM(nu=anomal_ratio, kernel = 'rbf', gamma='auto')
ocsvm_model.fit(ad_train_data)

#%% 모델 저장
scaler_fname = 'OCSVM_MinmaxScaler.pkl'
with open(os.path.join(work_dir, scaler_fname), 'wb') as f:
    pickle.dump(scaler, f)

encoder_fname = 'coupang_encoder_ocsvm.h5'
encoder_model.save(os.path.join(work_dir, encoder_fname))

ocsvm_fname = 'coupang_ocsvm.pkl'
with open(os.path.join(work_dir, ocsvm_fname), 'wb') as f:
    pickle.dump(ocsvm_model, f)
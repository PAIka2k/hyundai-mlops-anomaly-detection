# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pickle
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

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
           'msr_tbmax_raw', 'msr_tbmin_raw', 'SOC', 'CF_OBC_DCChargingStat', 'chg_charging_now']

# %% 데이터 로딩
work_dir = './'
fname = 'pu_batt_sample10.csv'
data = pd.read_csv(os.path.join(work_dir, fname), header=0, usecols=signals)
data = data[signals]

data = data[(data['CF_OBC_DCChargingStat'] == 1) & (data['chg_charging_now'] == 1)]
data['delta_voltage'] = data.apply(lambda x: x['msr_data.vb_max'] - x['msr_data.vb_min'], axis=1)
data['delta_temp'] = data.apply(lambda x: x['msr_tbmax_raw'] - x['msr_tbmin_raw'], axis=1)

data.drop(['CF_OBC_DCChargingStat', 'chg_charging_now'], axis=1, inplace=True)

scaler_fname = 'OCSVM_MinmaxScaler.pkl'
with open(os.path.join(work_dir, scaler_fname), 'rb') as f:
    scaler = pickle.load(f)

scaled_data = scaler.transform(data)
data = pd.DataFrame(scaled_data, columns=data.columns)

# 전셀전압
vol_signals = [signal for signal in signals if signal.startswith('cell_')]
vol_data = data[vol_signals]
vol_data = np.array(vol_data)

# 저항, 전류, 온도 등 전셀전압이외 데이터
other_data = data.drop(columns=vol_signals)

# 차원축소 모델 불러오기
encoder_fname = 'coupang_encoder_ocsvm.h5'
encoder_model = tf.keras.models.load_model(os.path.join(work_dir, encoder_fname))

with tf.device('/device:GPU:0'):
    # 입력 데이터에 대한 latent vector 추출
    latent_vector = encoder_model.predict(vol_data)

# %%
df_latent_vector = pd.DataFrame(latent_vector)

# %%
ad_test_data = pd.concat([other_data, df_latent_vector], axis=1)

# %% 모델 로딩
ocsvm_fname = 'coupang_ocsvm.pkl'
with open(os.path.join(work_dir, ocsvm_fname), 'rb') as f:
    ocsvm = pickle.load(f)

# %% 모델 테스트

# 이상탐지 결과
ocsvm_result = ocsvm.predict(ad_test_data)
ad_test_data = np.array(ad_test_data)
plt.plot(range(len(ad_test_data)), ad_test_data[:, 8], color='b', label='Normal')
outliers = ad_test_data[ocsvm_result == -1]
plt.scatter(np.where(ocsvm_result == -1)[0], outliers[:, 8], color='r', marker='o', label='Anomaly')
plt.xlabel('smaples')
plt.ylabel('delta_voltage')
plt.title('Outlier Detection')
plt.legend()
plt.show()

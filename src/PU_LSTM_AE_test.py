# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pickle
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
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

scaler_fname = 'LSTM_AE_MinmaxScaler.pkl'
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
encoder_fname = 'LSTM_AE_Dimension_Reduction_Model.h5'
encoder_model = tf.keras.models.load_model(os.path.join(work_dir, encoder_fname))
with tf.device('/device:GPU:0'):
    # 입력 데이터에 대한 latent vector 추출
    latent_vector = encoder_model.predict(vol_data)

# %%
df_latent_vector = pd.DataFrame(latent_vector)

# %%
ad_test_data = pd.concat([other_data, df_latent_vector], axis=1)

# %% 입력데이터 윈도윙
window_size = 200
stride = 100
windowed_data = []

for i in range(int(max(ad_test_data['cycle_num'])) + 1):
    window1 = ad_test_data.loc[ad_test_data['cycle_num'] == i]

    for j in range(0, len(window1) - window_size, stride):
        window2 = window1.iloc[j:j + window_size]
        windowed_data.append(window2)

np_windows = np.array(windowed_data)
# cycle_num 컬럼 삭제
test_data = np.delete(np_windows, [9], axis=2)

# %%
# 모델 불러오기
autoencoder = tf.keras.models.load_model('LSTM_AE_Model.h5')

# %%

# 테스트 데이터에 대한 예측 수행
decoded_original = autoencoder.predict(test_data)
mse = np.mean(np.square(test_data - decoded_original), axis=(1, 2))

# %%
# Reconstruction error를 그래프로 그리기
plt.plot(mse)
plt.xlabel('Sample Index')
plt.ylabel('Reconstruction Error')
plt.title('Reconstruction Error of Test Data')
plt.show()

import datetime
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class PreprocessData():
    def __init__(self,
                 dataset_path,
                 model_name):
        self.path = dataset_path
        self.model_name = model_name
        self.signals = self.target_signals(model_name)

    def preprocess_data(self):
        data = pd.read_csv(self.path, header=0, usecols=self.signals)
        data = data[self.signals]

        data = data[(data['CF_OBC_DCChargingStat'] == 1) & (data['chg_charging_now'] == 1)]
        data['delta_voltage'] = data.apply(lambda x: x['msr_data.vb_max'] - x['msr_data.vb_min'], axis=1)
        data['delta_temp'] = data.apply(lambda x: x['msr_tbmax_raw'] - x['msr_tbmin_raw'], axis=1)

        data.drop(['CF_OBC_DCChargingStat', 'chg_charging_now'], axis=1, inplace=True)

        if self.model_name == 'lstm':
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
        vol_signals = [signal for signal in self.signals if signal.startswith('cell_')]
        scaled_vol_data = scaled_data[vol_signals]

        # 저항, 전류, 온도 등 전셀전압이외 데이터
        scaled_other_data = scaled_data.drop(columns=vol_signals)
        return scaled_vol_data, scaled_other_data

    @staticmethod
    def target_signals(model_name):
        base_signals = ['cell_01', 'cell_02', 'cell_03', 'cell_04', 'cell_05', 'cell_06',
                        'cell_07', 'cell_08', 'cell_09', 'cell_10', 'cell_11', 'cell_12',
                        'cell_13', 'cell_14', 'cell_15', 'cell_16', 'cell_17', 'cell_18',
                        'cell_19', 'cell_20', 'cell_21', 'cell_22', 'cell_23', 'cell_24',
                        'cell_25', 'cell_26', 'cell_27', 'cell_28', 'cell_29', 'cell_30',
                        'cell_31', 'cell_32', 'cell_33', 'cell_34', 'cell_35', 'cell_36',
                        'cell_37', 'cell_38', 'cell_39', 'cell_40', 'cell_41', 'cell_42',
                        'cell_43', 'cell_44', 'cell_45', 'cell_46', 'cell_47', 'cell_48',
                        'cell_49', 'cell_50', 'cell_51', 'cell_52', 'cell_53', 'cell_54',
                        'cell_55', 'cell_56', 'cell_57', 'cell_58', 'cell_59', 'cell_60',
                        'cell_61', 'cell_62', 'cell_63', 'cell_64', 'cell_65', 'cell_66',
                        'cell_67', 'cell_68', 'cell_69', 'cell_70', 'cell_71', 'cell_72',
                        'cell_73', 'cell_74', 'cell_75', 'cell_76', 'cell_77', 'cell_78',
                        'cell_79', 'cell_80', 'cell_81', 'cell_82', 'cell_83', 'cell_84',
                        'cell_85', 'cell_86', 'cell_87', 'cell_88', 'cell_89', 'cell_90',
                        'msr_data.ibm', 'msr_data.r_isol', 'msr_data.vb_max', 'msr_data.vb_min',
                        'msr_tbmax_raw', 'msr_tbmin_raw', 'SOC', 'CF_OBC_DCChargingStat', 'chg_charging_now']
        if model_name == 'ocsvm':
            return base_signals
        elif model_name == 'lstm':
            base_signals.extend(['dnt', 'drv_cyc'])
            return base_signals

import boto3
import os
import io
###
# from boto3.s3.transfer import TransferConfig
# from botocore.exceptions import ClientError
import calendar
import pandas as pd
import struct
import asammdf
from asammdf import MDF
import numpy as np
import datetime as dt
from datetime import datetime
import re
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def init_s3_client(endpoint, access_key_id, secret_access_key):
    """
    :param endpoint: http://krcloud.s3.hmckmc.co.kr
    :type endpoint: str
    :param access_key_id: ACCESS-KEY-ID
    :type access_key_id: str
    :param secret_access_key: SECRET-ACCESS-KEY
    :type secret_access_key: str
    :return: boto3 S3 Client
    :rtype: S3.Client
    """    
    return boto3.client('s3',
                        endpoint_url=endpoint,
                        aws_access_key_id=access_key_id,
                        aws_secret_access_key=secret_access_key,
                        verify=False)

def get_s3_vdms_objs(s3_client, bucket_name, vhcl_key_ids, date_range, data_tp):
    def get_last_day_of_month(year, month):
        _, last_day = calendar.monthrange(year, month)
        return last_day
    
    # Search data by year
    years = [date_range[0][0:4], date_range[1][0:4]]
    years = list(set(years))
    years.sort()

    # Set arguments by lower case
    for i, elem in enumerate(data_tp):
        data_tp[i] = elem.lower()
    
    # Get object list
    paginator = s3_client.get_paginator('list_objects_v2')
    obj_list = []
    obj_summary = []
    if 'can' in data_tp or 'ccp' in data_tp:
        for key_id in vhcl_key_ids:
            print('Searching CAN/CCP data for key_id:',key_id)
            for year in years:
                for i in range(1,13):
                    for j in range(1, get_last_day_of_month(int(year), i)+1):
                        bucket_prefix='user/vdmsapp/merge_auto/mdf/'+ year +'/{0}/{1}/{2}'.format(f'{i:02d}',f'{j:02d}', key_id)
                        for page in paginator.paginate(Bucket=bucket_name, Prefix=bucket_prefix):
                            if 'Contents' in page:
                                for content in page['Contents']:
                                    if 'can' in content['Key'].lower() and 'can' in data_tp:
                                        obj_list.append([content['Key'], content['Size']])
                                    elif 'ccp' in content['Key'].lower() and 'ccp' in data_tp:
                                        obj_list.append([content['Key'], content['Size']])
                                    else:
                                        continue
    if 'gps' in data_tp or 'diag' in data_tp:
        for key_id in vhcl_key_ids:
            print('Searching GPS/DIAG data for key_id: ',key_id)
            for year in years:
                for i in range(1,13):
                    for j in range(1, get_last_day_of_month(int(year), i)+1):
                        bucket_prefix='user/vdmsapp/merge_auto/raw/'+ year +'/{0}/{1}/{2}'.format(f'{i:02d}',f'{j:02d}', key_id)
                        for page in paginator.paginate(Bucket=bucket_name, Prefix=bucket_prefix):
                            if 'Contents' in page:
                                for content in page['Contents']:
                                    if 'gps' in content['Key'].lower() and 'gps' in data_tp:
                                        obj_list.append([content['Key'], content['Size']])
                                    elif 'diag' in content['Key'].lower() and 'diag' in data_tp:
                                        obj_list.append([content['Key'], content['Size']])
                                    else:
                                        continue
    if len(data_tp)==0:
        print('Need to set VDMS data type to read')
    obj_list = pd.DataFrame(obj_list, columns=['Key', 'Size'])
    
    # Filter data by date range
    out_obj_list = []
    obj_summary = []
    for obj, obj_size in zip(obj_list['Key'], obj_list['Size']):
        fname = obj.split('/')[-1]
        can_cnt = 0
        ccp_cnt = 0
        gps_cnt = 0
        diag_cnt = 0
        if fname.lower().endswith('.dat'):
            temp1 = fname.lower().split('.')[0].split('_')
            temp2 = []
            temp_date = []
            for j in temp1:
                if j.isdigit() == True:
                    if int(j) > 2e13:
                        temp_date.append(j)
                    else:
                        temp2.append(j)
            key_id = temp2[-1]
            if len(temp_date) > 1:
                temp_date = int(temp_date[-1][0:8])
            else:
                temp_date = int(temp_date[0][0:8])
            if temp_date >= int(date_range[0]) and temp_date <= int(date_range[1]):
                out_obj_list.append([obj, obj_size])
                if 'can' in temp1:
                    can_cnt = can_cnt + 1
                elif 'ccp' in temp1:
                    ccp_cnt = ccp_cnt + 1
                elif 'gps' in temp1:
                    gps_cnt = gps_cnt + 1
                elif 'diag' in temp1:
                    diag_cnt = diag_cnt + 1
                else:
                    print('Check error data: ',obj)
                obj_summary.append([key_id, can_cnt, ccp_cnt, gps_cnt, diag_cnt])
        else:
            continue

    # Create output object list
    out_obj_list = pd.DataFrame(out_obj_list, columns=['Key','Size'])
    
    # Create summary of object list
    obj_summary = pd.DataFrame(obj_summary, columns=['key_id', 'can_cnt', 'ccp_cnt', 'gps_cnt', 'diag_cnt'])
    obj_summary  = obj_summary.groupby(['key_id']).sum()
    print('Done.')
    print('\n')
    print('Summary of object list')
    print(obj_summary)

    return out_obj_list

def filter_s3_drvlog_objs(obj_list, data_tp, min_size):
    # Set arguments by lower case
    for i, elem in enumerate(data_tp):
        data_tp[i] = elem.lower()
    
    # Count pairs of drvlog data
    drvlog_info = []
    for obj in obj_list['Key']:
        fname = obj.split('/')[-1]
        temp1 = fname.lower().split('.')[0].split('_')
        if 'diag' not in temp1:
            can_cnt = 0
            ccp_cnt = 0
            gps_cnt = 0
            temp2 = []
            for j in temp1:
                if j.isdigit() == True:
                    temp2.append(j)
            drv_cyc = temp2[0]
            key_id = temp2[-1]
            if 'can' in temp1:
                can_cnt = can_cnt + 1
            elif 'ccp' in temp1:
                ccp_cnt = ccp_cnt + 1
            elif 'gps' in temp1:
                gps_cnt = gps_cnt + 1
            temp3 = key_id, drv_cyc, can_cnt, ccp_cnt, gps_cnt
            drvlog_info.append(temp3)

    data_tp_columns = []
    if 'can' in data_tp:
        data_tp_columns.append('can_cnt')
    if 'ccp' in data_tp:
        data_tp_columns.append('ccp_cnt')
    if 'gps' in data_tp:
        data_tp_columns.append('gps_cnt')
    all_columns = ['key_id', 'drv_cyc', 'can_cnt', 'ccp_cnt', 'gps_cnt']
    drvlog_info = pd.DataFrame(drvlog_info, columns=all_columns)
    drvlog_sum = drvlog_info.groupby(all_columns[0:2]).sum()
    drvlog_sum = drvlog_sum[data_tp_columns]

    # Extract error dat files such as double counting or missing files that should be excluded before parsing
    # Note that default option is automatically excluding error files, which may lose data you want to investigate on
    cond_list = []
    for i in data_tp_columns:
        if i != data_tp_columns[-1]:
            temp_str = '(drvlog_sum["'+i+'"] != 1) |'
            cond_list.append(temp_str)
        else:
            temp_str = '(drvlog_sum["'+i+'"] != 1)'
            cond_list.append(temp_str)
    data_tp_cond = ' '.join(cond_list)
    drvlog_err = drvlog_sum.loc[eval(data_tp_cond)]
    print('Error data check:')
    if len(drvlog_err) == 0:
        print('No error data has been found\n')
    else:
        print(drvlog_err)
        print('\n')
    norm_drv1 = drvlog_sum.drop(drvlog_err.index)
    norm_drv1 = pd.DataFrame(list(norm_drv1.index), columns=['key_id', 'drv_cyc'])
    
    if 'can' in data_tp or 'ccp' in data_tp:
        #Filtering out small size data with corresponding drv_cyc
        th_size = min_size*1024**2 #MB
        temp4 = []
        for obj, obj_size in zip(obj_list['Key'], obj_list['Size']):
            fname = obj.split('/')[-1]
            temp1 = fname.lower().split('.')[0].split('_')
            can_cnt = 0
            ccp_cnt = 0
            if ('can' in temp1 or 'ccp' in temp1) and (obj_size) > th_size:
                temp2 = []
                for j in temp1:
                    if j.isdigit() == True:
                        temp2.append(j)
                drv_cyc = temp2[0]
                key_id = temp2[-1]
                if 'can' in temp1:
                    can_cnt = can_cnt + 1
                elif 'ccp' in temp1:
                    ccp_cnt = ccp_cnt + 1
                temp3 = key_id, drv_cyc, can_cnt, ccp_cnt
                temp4.append(temp3)
        drvlog_info2 = pd.DataFrame(temp4, columns=all_columns[:-1])
        drvlog_sum2 = drvlog_info2.groupby(all_columns[0:2]).sum()
        if 'gps_cnt' in data_tp_columns:
            data_tp_columns.remove('gps_cnt')
        else:
            data_tp_columns = data_tp_columns
        cond_list = []
        for i in data_tp_columns:
            if i != data_tp_columns[-1]:
                temp_str = '(drvlog_sum2["'+i+'"] != 1) |'
                cond_list.append(temp_str)
            else:
                temp_str = '(drvlog_sum2["'+i+'"] != 1)'
                cond_list.append(temp_str)
        data_tp_cond = ' '.join(cond_list)
        drvlog_err2 = drvlog_sum2.loc[eval(data_tp_cond)]
        norm_drv2 = drvlog_sum2.drop(drvlog_err2.index)
        norm_drv2 = pd.DataFrame(list(norm_drv2.index), columns=['key_id', 'drv_cyc'])
        norm_drv_mrg = pd.merge(norm_drv1, norm_drv2)
    else:
        norm_drv_mrg = norm_drv1.copy()
        
    # Get filtered objects
    temp3 = []
    for obj, obj_size in zip(obj_list['Key'], obj_list['Size']):
        fname = obj.split('/')[-1]
        temp1 = fname.lower().split('.')[0].split('_')
        for m in data_tp:
            if m in temp1:
                temp2 = []
                for j in temp1:
                    if j.isdigit() == True:
                        temp2.append(j)
                drv_cyc = temp2[0]
                key_id = temp2[-1]
                if len(norm_drv_mrg[(norm_drv_mrg['key_id'] == str(key_id)) & (norm_drv_mrg['drv_cyc'] == str(drv_cyc))]) > 0:
                    temp3.append([obj, obj_size])
    filt_obj_list = pd.DataFrame(temp3, columns=['Key', 'Size'])

    return filt_obj_list

class VDMSRAW:
    def __init__(self, filename):
        self.VehicleKey = 0
        self.PolicyVersion = 0
        self.RecordCnt = 0
        self.SN = 0
        self.BaseTime = 0
        self.MessageType = 0
        self.preTimeStamp = 0
        try:
            self.InFile = open(filename, 'rb', 1024)
        except:
            print("File Read Error!!")
        self.parseHeader()
    def parseHeader(self):
        header = self.InFile.read(26)
        self.VehicleKey, self.PolicyVersion, self.RecordCnt, self.SN, self.BaseTime, \
        self.MessageType =  struct.unpack('!IHI11sIB',header)
    def getMSG(self):
        dlc_Size = [0,1,2,3,4,5,6,7,8,12,16,20,24,32,48,64]
        try:
            fdata = self.InFile.read(1)
            DLC = fdata[0]
        except:
            return None
        #MSGInfo = ""
        fdata = self.InFile.read(10+dlc_Size[DLC])
        DeltaTime, DataFlag, DataChannel, DataID = struct.unpack('!IBBI',fdata[0:10])
        DeltaTime = DeltaTime*0.00005  # 1 tick is 50us
        if DataFlag == 2:
            MSGInfo = "Error Frame"
        else:
            if (DataFlag and 1) == 1:
                MSGInfo = "Extended ID"
            else:
                MSGInfo = "Standard ID"
            if (DataFlag and 4) == 4:
                MSGInfo = MSGInfo + " FD"
        return [self.BaseTime, DataChannel, DeltaTime, MSGInfo, DataID, DLC,
                fdata[10:10+dlc_Size[DLC]]]

    def getMSGInterval(self, interval):
        msg = self.getMSG()
        if msg == None:
            return None
        while (msg[2] - self.preTimeStamp) < interval:
            msg = self.getMSG()
            if msg == None:
                return None
        self.preTimeStamp = msg[2]
        return msg

class VDMSS3:
    def __init__(self, obj):
        self.VehicleKey = 0
        self.PolicyVersion = 0
        self.RecordCnt = 0
        self.SN = 0
        self.BaseTime = 0
        self.MessageType = 0
        self.preTimeStamp = 0
        try:
            self.InFile = obj
        except:
            print("File Read Error!!")
        self.parseHeader()
    def parseHeader(self):
        header = self.InFile.read(26)
        self.VehicleKey, self.PolicyVersion, self.RecordCnt, self.SN, self.BaseTime, \
        self.MessageType =  struct.unpack('!IHI11sIB',header)
    def getMSG(self):
        dlc_Size = [0,1,2,3,4,5,6,7,8,12,16,20,24,32,48,64]
        try:
            fdata = self.InFile.read(1)
            DLC = fdata[0]
        except:
            return None
        #MSGInfo = ""
        fdata = self.InFile.read(10+dlc_Size[DLC])
        DeltaTime, DataFlag, DataChannel, DataID = struct.unpack('!IBBI',fdata[0:10])
        DeltaTime = DeltaTime*0.00005  # 1 tick is 50us
        if DataFlag == 2:
            MSGInfo = "Error Frame"
        else:
            if (DataFlag and 1) == 1:
                MSGInfo = "Extended ID"
            else:
                MSGInfo = "Standard ID"
            if (DataFlag and 4) == 4:
                MSGInfo = MSGInfo + " FD"
        return [self.BaseTime, DataChannel, DeltaTime, MSGInfo, DataID, DLC,
                fdata[10:10+dlc_Size[DLC]]]

    def getMSGInterval(self, interval):
        msg = self.getMSG()
        if msg == None:
            return None
        while (msg[2] - self.preTimeStamp) < interval:
            msg = self.getMSG()
            if msg == None:
                return None
        self.preTimeStamp = msg[2]
        return msg

def vdms_mdfparsing(data_tp, can_chlist_file, mdf_file, samp_period, time_from_zero, s3_use, s3_client, bucket_name):
    # Parsing CAN mdf file
    if data_tp.lower() == 'can':
        # If user defined the channel list
        if len(can_chlist_file) != 0:
            chlist = pd.read_csv(can_chlist_file, sep=',', header=0, engine='python')
            # print('Target CAN channel list:\n', chlist)
            # print('\n')
            nan_check1 = chlist.iloc[:, 0].isnull().sum()  # ch_name
            nan_check2 = chlist.iloc[:, 2].isnull().sum()  # ch_id
            nan_check3 = chlist.iloc[:, 1].isnull().sum()  # msg_id
            if s3_use == True:
                temp_obj = s3_client.get_object(Bucket=bucket_name, Key=mdf_file)
                temp_obj = temp_obj['Body'].read()
                can_obj = io.BytesIO(temp_obj)
                # MDF size 큰 경우 try, except
                try:
                    mdf_read = MDF(can_obj)
                except: 
                    return [], [], []
            else:
                # MDF size 큰 경우 try, except
                try:
                    mdf_read = MDF(mdf_file)
                except: 
                    return [], [], []
            mdf_info = mdf_read.info()
            # print('Parsing CAN mdf data....',end=' ')
            
            out_list = []
            for p in range(0, len(chlist)):  # Number of channels that you want to search for
                for i in range(0, len(mdf_read.groups)):  # Number of channel groups in mdf file
                    temp_grp_idx = 'group ' + str(i)
                    temp_cycles = int(mdf_info[temp_grp_idx]['cycles'])
                    # Check error data (cycles = 0)
                    if temp_cycles == 0:
                        continue
                    else:
                        temp_cmnt = mdf_info[temp_grp_idx]['comment'].split(',')
                        temp_ch_id = int(re.sub(r'[^0-9]', '', temp_cmnt[0].split(':')[1]))
                        temp_msg_id = int(re.sub(r'[^0-9]', '', temp_cmnt[1].split(':')[1]))
                        # temp_msg_name = re.sub(r'[^\uAC00-\uD7A30-9a-zA-Z\s]', '', temp_cmnt[2].split(':')[1])
                        temp_msg_name = re.sub(r'[\\"\s]', '', temp_cmnt[2].split(':')[1])
                        temp_ch_cnt = int(mdf_info[temp_grp_idx]['channels count'])
                        temp_cycles = int(mdf_info[temp_grp_idx]['cycles'])
                        # Defined all parameters in target channel list: [ch_name, msg_id, ch_id]
                        if nan_check1 + nan_check2 + nan_check3 == 0:
                            srch_ch_name = chlist.iloc[p, 0]
                            srch_ch_id = int(chlist.iloc[p, 2])
                            srch_msg_id = int(chlist.iloc[p, 1])
                            if (temp_msg_id == srch_msg_id) & (temp_ch_id == srch_ch_id):
                                for j in range(0, temp_ch_cnt):
                                    temp_ch_idx = 'channel ' + str(j)
                                    temp_ch_name = mdf_info[temp_grp_idx][temp_ch_idx].split('"')[1]
                                    if temp_ch_name.lower() == srch_ch_name.lower():
                                        temp_out_list = temp_ch_name, temp_ch_id, temp_msg_name, temp_msg_id, i, j, temp_cycles
                                        out_list.append(temp_out_list)

                        # Defined two parameters in target channel list: [ch_name, ch_id]
                        if (nan_check2 == 0) & (nan_check3 != 0):
                            srch_ch_name = chlist.iloc[p, 0]
                            srch_ch_id = int(chlist.iloc[p, 2])
                            if temp_ch_id == srch_ch_id:
                                for j in range(0, temp_ch_cnt):
                                    temp_ch_idx = 'channel ' + str(j)
                                    temp_ch_name = mdf_info[temp_grp_idx][temp_ch_idx].split('"')[1]
                                    if temp_ch_name.lower() == srch_ch_name.lower():
                                        temp_out_list = temp_ch_name, temp_ch_id, temp_msg_name, temp_msg_id, i, j, temp_cycles
                                        out_list.append(temp_out_list)

                        # Defined two parameters in target channel list: [ch_name, msg_id]
                        if (nan_check2 != 0) & (nan_check3 == 0):
                            srch_ch_name = chlist.iloc[p, 0]
                            srch_msg_id = int(chlist.iloc[p, 1])
                            if temp_msg_id == srch_msg_id:
                                for j in range(0, temp_ch_cnt):
                                    temp_ch_idx = 'channel ' + str(j)
                                    temp_ch_name = mdf_info[temp_grp_idx][temp_ch_idx].split('"')[1]
                                    if temp_ch_name.lower() == srch_ch_name.lower():
                                        temp_out_list = temp_ch_name, temp_ch_id, temp_msg_name, temp_msg_id, i, j, temp_cycles
                                        out_list.append(temp_out_list)

                        # Defined only ch_name in target channel list
                        if nan_check2 * nan_check3 != 0:
                            srch_ch_name = chlist.iloc[p, 0]
                            for j in range(0, temp_ch_cnt):
                                temp_ch_idx = 'channel ' + str(j)
                                temp_ch_name = mdf_info[temp_grp_idx][temp_ch_idx].split('"')[1]
                                if temp_ch_name.lower() == srch_ch_name.lower():
                                    temp_out_list = temp_ch_name, temp_ch_id, temp_msg_name, temp_msg_id, i, j, temp_cycles
                                    out_list.append(temp_out_list)

            temp1 = pd.DataFrame(out_list,
                                 columns=['ch_name', 'ch_id', 'msg_name', 'msg_id', 'grp_idx', 'ch_idx', 'cycles'])
            temp2 = temp1.sort_values(by=['ch_name', 'cycles'], ascending=[True, False])
            out_list_uniq = temp2.drop_duplicates(['ch_name'], keep='first').reset_index(drop=True)
            out_list_uniq.insert(0, 'data_tp', 'CAN')
            temp3 = out_list_uniq[['ch_name', 'grp_idx', 'ch_idx']].values.tolist()
            mdf_out = mdf_read.filter(temp3)
            prsd_df = mdf_out.to_dataframe(raster=samp_period, time_from_zero=time_from_zero,
                                           reduce_memory_usage=True, ignore_value2text_conversions=True)
            col_prsd_df = list(prsd_df.columns)
            col_prsd_df.sort()
            ref_ch_df = pd.DataFrame(columns=chlist['ch_name'])
            prsd_df = pd.concat([ref_ch_df, prsd_df[col_prsd_df]], join='outer')

        # If user didn't define the channel list
        else:
            # print('None of channel has been selected. Please check your CAN channel list input file.')
            if s3_use == True:
                temp_obj = s3_client.get_object(Bucket=bucket_name, Key=mdf_file)
                temp_obj = temp_obj['Body'].read()
                can_obj = io.BytesIO(temp_obj)
                # MDF size 큰 경우 try, except
                try:
                    mdf_read = MDF(can_obj)
                except: 
                    return [], [], []
            else:
                # MDF size 큰 경우 try, except
                try:
                    mdf_read = MDF(mdf_file)
                except: 
                    return [], [], []
            mdf_info = mdf_read.info()

            # print('Parsing CAN mdf data....',end=' ')
            out_list = []
            for i in range(0, len(mdf_read.groups)):  # Number of channel groups in mdf file
                temp_grp_idx = 'group ' + str(i)
                temp_cycles = int(mdf_info[temp_grp_idx]['cycles'])
                # Check error data (cycles = 0)
                if temp_cycles == 0:
                    continue
                else:
                    temp_cmnt = mdf_info[temp_grp_idx]['comment'].split(',')
                    temp_ch_id = int(re.sub(r'[^0-9]', '', temp_cmnt[0].split(':')[1]))
                    temp_msg_id = int(re.sub(r'[^0-9]', '', temp_cmnt[1].split(':')[1]))
                    # temp_msg_name = re.sub(r'[^\uAC00-\uD7A30-9a-zA-Z\s]', '', temp_cmnt[2].split(':')[1])
                    temp_msg_name = re.sub(r'[\\"\s]', '', temp_cmnt[2].split(':')[1])
                    temp_ch_cnt = int(mdf_info[temp_grp_idx]['channels count'])
                    for j in range(1, temp_ch_cnt):
                        temp_ch_idx = 'channel ' + str(j)
                        temp_ch_name = mdf_info[temp_grp_idx][temp_ch_idx].split('"')[1]
                        temp_out_list = temp_ch_name, temp_ch_id, temp_msg_name, temp_msg_id, i, j, temp_cycles
                        out_list.append(temp_out_list)
            out_list_uniq = pd.DataFrame(out_list,
                                         columns=['ch_name', 'ch_id', 'msg_name', 'msg_id', 'grp_idx', 'ch_idx',
                                                  'cycles'])
            out_list_uniq.insert(0, 'data_tp', 'CAN')
            prsd_df = mdf_read.to_dataframe(raster=samp_period, time_from_zero=time_from_zero,
                                            reduce_memory_usage=True, ignore_value2text_conversions=True)
        # Add additional info
        if s3_use == True:
            temp = mdf_file.split('/')[-1]
            temp1 = temp.lower().split('.')[0].split('_')
        else:
            temp1 = mdf_file.lower().split('.')[0].split('_')
        temp_num = []
        temp_dnt = []
        temp_str = []
        for j in temp1:
            if j.isdigit() == True:
                if int(j) > 2e13:
                    temp_dnt.append(j)
                else:
                    temp_num.append(j)
            else:
                temp_str.append(j)
        vhcl_tp = temp_str[0]
        mng_id = temp_str[1]
        key_id = temp_num[-1]
        drv_cyc = temp_num[0]

        tmstmp = np.round(list(prsd_df.index), 4)
        prsd_df.insert(0, 'tmstmp', tmstmp)
        prsd_df_out = prsd_df.reset_index(drop=True)

        if len(temp_dnt) > 1:
            dnt_bgn = temp_dnt[-1]
        else:
            dnt_bgn = temp_dnt[0]
        dnt_temp1 = dt.datetime(int(dnt_bgn[:4]), int(dnt_bgn[4:6]), int(dnt_bgn[6:8]),
                                int(dnt_bgn[8:10]), int(dnt_bgn[10:12]), int(dnt_bgn[12:14]))
        dnt_bgn2 = dnt_temp1.timestamp()
        dnt_end = dnt_bgn2 + (len(prsd_df_out) - 1) * samp_period
        dnt_temp2 = list(np.linspace(dnt_bgn2, dnt_end, len(prsd_df_out)))
        dnt = []
        for ii in range(0, len(dnt_temp2)):
            dnt_temp3 = dt.datetime.fromtimestamp(dnt_temp2[ii])
            dnt.append(dnt_temp3)

        fid = key_id + '_' + str(drv_cyc) + '_' + str(dnt_bgn)
        prsd_df_out.insert(0, 'drv_cyc', drv_cyc)
        prsd_df_out.insert(0, 'key_id', key_id)
        prsd_df_out.insert(0, 'mng_id', mng_id)
        prsd_df_out.insert(0, 'vhcl_tp', vhcl_tp)
        prsd_df_out.insert(0, 'dnt', dnt)
        prsd_df_out.insert(0, 'fid', fid)
        out_list_uniq.insert(0, 'fid', fid)
        out_mdf_flist = fid, mdf_file
        # print('Done.')
        return prsd_df_out, out_list_uniq, out_mdf_flist

    if data_tp.lower() == 'ccp':
        can_chlist_file = []
        if s3_use == True:
            temp_obj = s3_client.get_object(Bucket=bucket_name, Key=mdf_file)
            temp_obj = temp_obj['Body'].read()
            can_obj = io.BytesIO(temp_obj)
            # MDF size 큰 경우 try, except
            try:
                mdf_read = MDF(can_obj)
            except: 
                return [], [], []
        else:
            # MDF size 큰 경우 try, except
            try:
                mdf_read = MDF(mdf_file)
            except: 
                return [], [], []
        mdf_info = mdf_read.info()

        # print('Parsing CCP mdf data....', end=' ')
        # check error data (cycles = 0)
        temp_cycles = int(mdf_info['group 0']['cycles'])
        
        if temp_cycles == 0:
            print('Error data')
        else:
            temp_ch_cnt = int(mdf_info['group 0']['channels count'])
            out_list = []
            for j in range(1, temp_ch_cnt):
                temp_ch_idx = 'channel ' + str(j)
                temp_ch_name = mdf_info['group 0'][temp_ch_idx].split('"')[1]
                temp_out_list = temp_ch_name, j, temp_cycles
                out_list.append(temp_out_list)
            out_list_uniq = pd.DataFrame(out_list, columns=['ch_name','ch_idx','cycles'])
            out_list_uniq.insert(0, 'data_tp', 'CCP')
            prsd_df = mdf_read.to_dataframe(raster=samp_period, time_from_zero=time_from_zero,
                                            reduce_memory_usage=True, ignore_value2text_conversions=True)
            # Add additional info
            if s3_use == True:
                temp = mdf_file.split('/')[-1]
                temp1 = temp.lower().split('.')[0].split('_')
            else:
                temp1 = mdf_file.lower().split('.')[0].split('_')
            temp_num = []
            temp_dnt = []
            temp_str = []
            for j in temp1:
                if j.isdigit() == True:
                    if int(j) > 2e13:
                        temp_dnt.append(j)
                    else:
                        temp_num.append(j)
                else:
                    temp_str.append(j)
            vhcl_tp = temp_str[0]
            mng_id = temp_str[1]
            key_id = temp_num[-1]
            drv_cyc = temp_num[0]

            tmstmp = np.round(list(prsd_df.index), 4)
            prsd_df.insert(0, 'tmstmp', tmstmp)
            prsd_df_out = prsd_df.reset_index(drop=True)

            if len(temp_dnt) > 1:
                dnt_bgn = temp_dnt[-1]
            else:
                dnt_bgn = temp_dnt[0]
            dnt_temp1 = dt.datetime(int(dnt_bgn[:4]), int(dnt_bgn[4:6]), int(dnt_bgn[6:8]),
                                    int(dnt_bgn[8:10]), int(dnt_bgn[10:12]), int(dnt_bgn[12:14]))
            dnt_bgn2 = dnt_temp1.timestamp()
            dnt_end = dnt_bgn2 + (len(prsd_df_out) - 1) * samp_period
            dnt_temp2 = list(np.linspace(dnt_bgn2, dnt_end, len(prsd_df_out)))
            dnt = []
            for ii in range(0, len(dnt_temp2)):
                dnt_temp3 = dt.datetime.fromtimestamp(dnt_temp2[ii])
                dnt.append(dnt_temp3)

            fid = key_id + '_' + drv_cyc + '_' + dnt_bgn
            prsd_df_out.insert(0, 'drv_cyc', drv_cyc)
            prsd_df_out.insert(0, 'key_id', key_id)
            prsd_df_out.insert(0, 'mng_id', mng_id)
            prsd_df_out.insert(0, 'vhcl_tp', vhcl_tp)
            prsd_df_out.insert(0, 'dnt', dnt)
            prsd_df_out.insert(0, 'fid', fid)
            out_list_uniq.insert(0, 'fid', fid)
            out_mdf_flist = fid, mdf_file
            # print('Done.')
            return prsd_df_out, out_list_uniq, out_mdf_flist

def vdms_gpsparsing(gps_file, samp_period, s3_use, s3_client, bucket_name):
    def decodegps(msg):
        RawLat, RawLong, Heading, Speed, Altitude = struct.unpack('!IIHBH', msg[6][0:13])
        # dnt = datetime.utcfromtimestamp(msg[0] + msg[2] + 9 * 3600)
        Lat = ['', 0]
        Long = ['', 0]
        if RawLat & 0x80000000 == 0x80000000:
            Lat[0] = 'S'
        else:
            Lat[0] = 'N'
        Lat[1] = (RawLat & 0x7FFFFFFF) * 0.0000001
        if RawLong & 0x80000000 == 0x80000000:
            Long[0] = 'W'
        else:
            Long[0] = 'E'
        Long[1] = (RawLong & 0x7FFFFFFF) * 0.0000001
        # return [[dnt, msg[2], Lat[0], Lat[1], Long[0], Long[1], Heading / 10, Speed, Altitude]]
        return [[msg[2], Lat[0], Lat[1], Long[0], Long[1], Heading / 10, Speed, Altitude]]

    # print('Parsing GPS data....', end=' ')
    labels = ['tmstmp', 'Lat0', 'Lat1', 'Long0', 'Long1',
              'Heading', 'Speed', 'Altitude']
    if s3_use == True:
        temp_obj = s3_client.get_object(Bucket=bucket_name, Key=gps_file)
        temp_obj = temp_obj['Body'].read()
        gps_obj = io.BytesIO(temp_obj)
        rawdata = VDMSS3(gps_obj)
    else:
        rawdata = VDMSS3(gps_file)
    counter = 0
    while True:
        msg = rawdata.getMSGInterval(1)
        if msg == None:
            break
        elif counter == 0:
            parsedgps = np.array(decodegps(msg))
        else:
            parsedgps = np.append(parsedgps, decodegps(msg), axis=0)
        counter = counter + 1

    df_temp1 = pd.DataFrame(parsedgps)
    df_temp1.columns = labels

    # Add additional info
    if s3_use == True:
        temp = gps_file.split('/')[-1]
        temp1 = temp.lower().split('.')[0].split('_')
    else:
        temp1 = gps_file.lower().split('.')[0].split('_')
    temp_num = []
    temp_dnt = []
    temp_str = []
    for j in temp1:
        if j.isdigit() == True:
            if int(j) > 2e13:
                temp_dnt.append(j)
            else:
                temp_num.append(j)
        else:
            temp_str.append(j)
    vhcl_tp = temp_str[0]
    mng_id = temp_str[1]
    key_id = temp_num[-1]
    drv_cyc = temp_num[0]
    if len(temp_dnt) > 1:
        dnt_bgn = temp_dnt[-1]
    else:
        dnt_bgn = temp_dnt[0]
    dnt_temp1 = dt.datetime(int(dnt_bgn[:4]), int(dnt_bgn[4:6]), int(dnt_bgn[6:8]),
                            int(dnt_bgn[8:10]), int(dnt_bgn[10:12]), int(dnt_bgn[12:14]))
    dnt_bgn2 = dnt_temp1.timestamp()
    dnt_end = dnt_bgn2 + (counter - 1)
    dnt_temp2 = list(np.linspace(dnt_bgn2, dnt_end, counter))
    dnt = []
    for ii in range(0, len(dnt_temp2)):
        dnt_temp3 = dt.datetime.fromtimestamp(dnt_temp2[ii])
        dnt.append(dnt_temp3)

    df_temp1.insert(0, 'dnt', dnt)
    df_temp2 = df_temp1.set_index('dnt')
    if samp_period == 1:
        res_period = '1S'
    elif samp_period == 0.1:
        res_period = '100L'
    else:
        print('Need to select 0.1s or 1s of resampling period.')
    df_temp3 = df_temp2.resample(res_period).nearest()
    df_temp3.insert(0, 'dnt', df_temp3.index)
    df_gps = df_temp3.reset_index(drop=True)
    
    gps_fid = key_id + '_' + str(drv_cyc) + '_' + str(dnt_bgn)
    out_gps_flist = gps_fid, gps_file
    df_gps.insert(0, 'fid', gps_fid)
    out_list = pd.DataFrame()
    out_list.insert(0, 'ch_name', labels[1:])
    out_list.insert(0, 'data_tp', 'GPS')
    out_list.insert(0, 'fid', gps_fid)
    out_list['cycles'] = counter
    # print('Done.')
    return df_gps, out_list, out_gps_flist

def s3_vdms_pars_merging(obj_list, data_tp, can_chlist_file, samp_period, s3_client, save_dir):
    # Set arguments by lower case
    for i, elem in enumerate(data_tp):
        data_tp[i] = elem.lower()

    # Get key_id and drv_cyc info from obj list
    temp_info = []
    for obj in obj_list['Key']:
        fname = obj.split('/')[-1]
        temp1 = fname.lower().split('.')[0].split('_')
        temp2 = []
        for j in temp1:
            if j.isdigit() == True:
                temp2.append(j)
        drv_cyc = temp2[0]
        key_id = temp2[-1]
        temp_info.append([key_id, drv_cyc])
    temp_info = pd.DataFrame(temp_info, columns=['key_id', 'drv_cyc'])
    drvlog_info = temp_info.drop_duplicates(['key_id', 'drv_cyc'], keep='first').reset_index(drop=True)
    # Sort key_id, drv_cyc by ascending order
    drvlog_info = drvlog_info.sort_values(by=['key_id', 'drv_cyc'], ascending=[True, True])

    # Parsing vdms data
    can_flist = []
    ccp_flist = []
    gps_flist = []
    print('Parsing vdms data and saving outputs....',end=' ')

    # mdf file size error check
    size_error_check = False
    for ii in tqdm(range(0,len(drvlog_info))):
        i = drvlog_info['key_id'].iloc[ii]
        k = drvlog_info['drv_cyc'].iloc[ii]
        df_can = []
        df_ccp = []
        df_gps = []
        for obj in obj_list['Key']:
            fname = obj.split('/')[-1]
            temp1 = fname.lower().split('.')[0].split('_')
            temp2 = []
            for j in temp1:
                if j.isdigit() == True:
                    temp2.append(j)
            drv_cyc = temp2[0]
            key_id = temp2[-1]
            if int(key_id) == int(i) and int(drv_cyc) == int(k):
                if 'can' in temp1:
                    df_can, can_outlist, can_finfo = vdms_mdfparsing('can', can_chlist_file, obj, samp_period, time_from_zero=False, s3_use=True, s3_client=s3_client, bucket_name='vdms')
                    # mdf size error
                    if len(df_can) == 0:
                        size_error_check = True
                        break
                    can_flist.append(can_finfo)
                elif 'ccp' in temp1:
                    df_ccp, ccp_outlist, ccp_finfo = vdms_mdfparsing('ccp', [], obj, samp_period, time_from_zero=False, s3_use=True, s3_client=s3_client, bucket_name='vdms')
                    # mdf size error
                    if len(df_ccp) == 0:
                        size_error_check = True
                        break
                    ccp_flist.append(ccp_finfo)
                elif 'gps' in temp1:
                    df_gps, gps_outlist, gps_finfo = vdms_gpsparsing(obj, samp_period, s3_use=True, s3_client=s3_client, bucket_name='vdms')
                    gps_flist.append(gps_finfo)
                else:
                    print('Need to check dat files in key_id: %s, drv_cyc: %s'%(key_id,drv_cyc))
                    continue
        # mdf size error
        if size_error_check :
            size_error_check = False
            print('MDF file error in key_id: %s, drv_cyc: %s'%(i,k))
            continue
        # Merge all channels into a single dataframe
        df_fact = []
        df_ref_ch = []
        
        # tmstmp 자리수 통일 및 interpolate
        def tmstmp_matching(df, samp_period):
            df['tmstmp'] = df['tmstmp'].astype(float) 
            if samp_period == 1: 
                df['tmstmp'] = df['tmstmp'].apply(lambda x : round(x, 0))
            elif samp_period == 0.1:
                df['tmstmp'] = df['tmstmp'].apply(lambda x : round(x, 1))
            df = df.drop_duplicates('tmstmp')
            return df

        if 'can' in data_tp and 'ccp' in data_tp and 'gps' in data_tp: # CAN, CCP, GPS all
            df_ccp = df_ccp.drop(['fid','dnt','vhcl_tp','mng_id','key_id','drv_cyc'], axis=1)
            df_gps = df_gps.drop(['fid','dnt'], axis=1)
            df_can = tmstmp_matching(df_can,samp_period)
            df_ccp = tmstmp_matching(df_ccp,samp_period)
            df_gps = tmstmp_matching(df_gps,samp_period)
            df_temp = pd.merge(left=df_can, right=df_ccp, how='inner', on='tmstmp')
            df_fact = pd.merge(left=df_temp, right=df_gps, how='left', on='tmstmp')
            df_fact = df_fact.fillna(method='ffill').fillna(method='bfill')
            df_ref_ch = pd.concat([can_outlist, ccp_outlist, gps_outlist])
        else:
            if 'ccp' not in data_tp:
                if 'gps' not in data_tp: # CAN only
                    df_fact = df_can
                    df_ref_ch = can_outlist
                elif 'can' not in data_tp: # GPS only
                    df_fact = df_gps
                    df_ref_ch = gps_outlist
                else: # CAN, GPS
                    df_can = tmstmp_matching(df_can,samp_period)
                    df_gps = tmstmp_matching(df_gps,samp_period)
                    df_gps = df_gps.drop(['fid','dnt'], axis=1)
                    df_fact = pd.merge(left=df_can, right=df_gps, how='left', on='tmstmp')
                    df_fact = df_fact.fillna(method='ffill').fillna(method='bfill')
                    df_ref_ch = pd.concat([can_outlist, gps_outlist])
            elif 'gps' not in data_tp:
                if 'can' not in data_tp: # CCP only
                    df_fact = df_ccp
                    df_ref_ch = ccp_outlist
                else: # CAN, CCP
                    df_ccp = tmstmp_matching(df_ccp,samp_period)
                    df_can = tmstmp_matching(df_can,samp_period)
                    df_ccp = df_ccp.drop(['fid','dnt', 'vhcl_tp', 'mng_id', 'key_id', 'drv_cyc'], axis=1)
                    df_fact = pd.merge(left=df_can, right=df_ccp, how='inner', on='tmstmp')
                    df_ref_ch = pd.concat([can_outlist, ccp_outlist])
            elif 'can' not in data_tp: # CCP, GPS
                df_ccp = tmstmp_matching(df_ccp,samp_period)
                df_gps = tmstmp_matching(df_gps,samp_period)
                df_gps = df_gps.drop(['fid','dnt'], axis=1)
                df_fact = pd.merge(left=df_ccp, right=df_gps, how='left', on='tmstmp')
                df_fact = df_fact.fillna(method='ffill').fillna(method='bfill')
                df_ref_ch = pd.concat([ccp_outlist, gps_outlist])
            else:
                print('Need to check dat files in key_id: %s, drv_cyc: %s'%(key_id,drv_cyc))
                continue

        # Save fact table data (vehicle info, all channels data)
        # print('Saving output files....',end=' ')
        fname1 = df_fact['fid'][0] + '_fact.csv'
        save_name1 = os.path.join(save_dir, fname1)
        df_fact.to_csv(save_name1, sep=',', encoding='utf-8', index=False)

        # Save ref table2 data (list of processed signals)
        fname2 = df_fact['fid'][0] + '_ref_ch.csv'
        save_name2 = os.path.join(save_dir, fname2)
        df_ref_ch.to_csv(save_name2, sep=',', encoding='utf-8', index=False)
        # print('Done.')
                    
    # Save ref table 1 data (fid, file names)
    fname3 = 'prsd_files_ref.csv'
    save_name3 = os.path.join(save_dir, fname3)
    if 'can' in data_tp and 'ccp' in data_tp and 'gps' in data_tp: # CAN, CCP, GPS all
        df_flist = pd.DataFrame(columns=['fid', 'can_file', 'ccp_file', 'gps_file'])
        df_flist['fid'] = np.array(can_flist)[:,0]
        df_flist['can_file'] = np.array(can_flist)[:,1]
        df_flist['ccp_file'] = np.array(ccp_flist)[:,1]
        df_flist['gps_file'] = np.array(gps_flist)[:,1]
        df_flist.to_csv(save_name3, sep=',', encoding='utf-8', index=False)
    else:
        if 'ccp' not in data_tp:
            if 'gps' not in data_tp: # CAN only
                df_flist = pd.DataFrame(columns=['fid', 'can_file', 'ccp_file', 'gps_file'])
                df_flist['fid'] = np.array(can_flist)[:, 0]
                df_flist['can_file'] = np.array(can_flist)[:, 1]
                df_flist.to_csv(save_name3, sep=',', encoding='utf-8', index=False)
            elif 'can' not in data_tp: # GPS only
                df_flist = pd.DataFrame(columns=['fid', 'can_file', 'ccp_file', 'gps_file'])
                df_flist['fid'] = np.array(can_flist)[:, 0]
                df_flist['gps_file'] = np.array(gps_flist)[:, 1]
                df_flist.to_csv(save_name3, sep=',', encoding='utf-8', index=False)
            else: # CAN, GPS
                df_flist = pd.DataFrame(columns=['fid', 'can_file', 'ccp_file', 'gps_file'])
                df_flist['fid'] = np.array(can_flist)[:, 0]
                df_flist['can_file'] = np.array(can_flist)[:, 1]
                df_flist['gps_file'] = np.array(gps_flist)[:, 1]
                df_flist.to_csv(save_name3, sep=',', encoding='utf-8', index=False)
        elif 'gps' not in data_tp:
            if 'can' not in data_tp: # CCP only
                df_flist = pd.DataFrame(columns=['fid', 'can_file', 'ccp_file', 'gps_file'])
                df_flist['fid'] = np.array(can_flist)[:, 0]
                df_flist['ccp_file'] = np.array(ccp_flist)[:, 1]
                df_flist.to_csv(save_name3, sep=',', encoding='utf-8', index=False)
            else: # CAN, CCP
                df_flist = pd.DataFrame(columns=['fid', 'can_file', 'ccp_file', 'gps_file'])
                df_flist['fid'] = np.array(can_flist)[:, 0]
                df_flist['can_file'] = np.array(can_flist)[:, 1]
                df_flist['ccp_file'] = np.array(ccp_flist)[:, 1]
                df_flist.to_csv(save_name3, sep=',', encoding='utf-8', index=False)
        elif 'can' not in data_tp: # CCP, GPS
            df_flist = pd.DataFrame(columns=['fid', 'can_file', 'ccp_file', 'gps_file'])
            df_flist['fid'] = np.array(can_flist)[:, 0]
            df_flist['ccp_file'] = np.array(ccp_flist)[:, 1]
            df_flist['gps_file'] = np.array(gps_flist)[:, 1]
            df_flist.to_csv(save_name3, sep=',', encoding='utf-8', index=False)
        else:
            print('Need to check data or input settings')
    print('Done.')
import pandas as pd
import numpy as np
import time
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import os


columns = ['Destination_Port', 'Flow_Duration', 'Total_Fwd_Packets',
            'Total_Backward_Packets', 'Total_Length_of_Fwd_Packets',
            'Total_Length_of_Bwd_Packets', 'Fwd_Packet_Length_Max',
            'Fwd_Packet_Length_Min', 'Fwd_Packet_Length_Mean',
            'Fwd_Packet_Length_Std', 'Bwd_Packet_Length_Max',
            'Bwd_Packet_Length_Min', 'Bwd_Packet_Length_Mean',
            'Bwd_Packet_Length_Std', 'Flow_Bytes/s', 'Flow_Packets/s',
            'Flow_IAT_Mean', 'Flow_IAT_Std', 'Flow_IAT_Max', 'Flow_IAT_Min',
            'Fwd_IAT_Total', 'Fwd_IAT_Mean', 'Fwd_IAT_Std', 'Fwd_IAT_Max',
            'Fwd_IAT_Min', 'Bwd_IAT_Total', 'Bwd_IAT_Mean', 'Bwd_IAT_Std',
            'Bwd_IAT_Max', 'Bwd_IAT_Min', 'Fwd_PSH_Flags', 'Bwd_PSH_Flags',
            'Fwd_URG_Flags', 'Bwd_URG_Flags', 'Fwd_Header_Length',
            'Bwd_Header_Length', 'Fwd_Packets/s', 'Bwd_Packets/s',
            'Min_Packet_Length', 'Max_Packet_Length', 'Packet_Length_Mean',
            'Packet _Length_Std', ' Packet_Length_Variance', 'FIN_Flag_Count',
            'SYN_Flag_Count', 'RST_Flag_Count', 'PSH_Flag_Count',
            'ACK_Flag_Count', 'URG_Flag_Count', 'CWE_Flag_Count',
            'ECE_Flag_Count', 'Down/Up_Ratio', 'Average_Packet_Size',
            'Avg_Fwd_Segment_Size', 'Avg_Bwd_Segment_Size',
            'Fwd_Header_Length.1', 'Fwd_Avg_Bytes/Bulk', '_Fwd_Avg_Packets/Bulk',
            'Fwd_Avg_Bulk_Rate', 'Bwd_Avg_Bytes/Bulk', 'Bwd_Avg_Packets/Bulk',
            'Bwd_Avg_Bulk_Rate', 'Subflow_Fwd_Packets', 'Subflow_Fwd_Bytes',
            'Subflow_Bwd_Packets', 'Subflow_Bwd_Bytes', 'Init_Win_bytes_forward',
            'Init_Win_bytes_backward', 'act_data_pkt_fwd',
            'min_seg_size_forward', 'Active_Mean', 'Active_Std', 'Active_Max',
            'Active Min', 'Idle_Mean', 'Idle_Std', 'Idle_Max', 'Idle_Min',
            'Label']
csvfile = '/root/data/fxg/csv'
def Data_preprocessing(csvfile=csvfile):


    # csvfile = "/root/data/fxg/csv"
    all_scv_list = os.listdir(csvfile)
    for single_csv in all_scv_list:
        single_data = pd.read_csv(os.path.join(csvfile, single_csv), header=1)
        if single_csv == all_scv_list[0]:
            data = single_data
        else:
            data = pd.concat([data, single_data], ignore_index=True)
    data.columns = columns
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)
    data.Label[data['Label'] != 'BENIGN'] = 1
    data.Label[data['Label'] == 'BENIGN'] = 0

    X = data.drop(['Label'], axis=1)
    labelencoder_x=LabelEncoder()
    for i in columns[:-1]:
        X[i]=labelencoder_x.fit_transform(X[i])
    y = data['Label'].astype(int)


    # 标准化
    transfer = MinMaxScaler(feature_range=(2, 3))
    X = transfer.fit_transform(X)

    # 降维
    transfer = VarianceThreshold(threshold=0.0)
    X = transfer.fit_transform(X)
    # 主成分分析
    transfer = PCA(n_components=0.9)
    X = transfer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=20)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_train, X_test, y_train, y_test

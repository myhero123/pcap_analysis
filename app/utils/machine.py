from threading import main_thread
from unittest import result

from sklearn import naive_bayes

# from .feature_engineering import Data_preprocessing
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import scikitplot as skplt
import os

re=[]
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

#KNN
from sklearn.model_selection import GridSearchCV
def KNN(train_X=X_train, train_Y=y_train, test_X=X_test, test_Y=y_test):
    # print('[KNN] train...')
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(train_X, train_Y)
    y_hat=knn.predict(test_X)

    #模型评估
    acc = accuracy_score(test_Y, y_hat)  #
    # t2 = time.time()
    # print('准确率:', acc)
    # print('总耗时:', t2 - t1, 'sec')
    # print('精确度:', accuracy_score(test_Y, y_hat))
    # print('混淆矩阵：\n',confusion_matrix(test_Y, y_hat))
    # print('KNN分类报告为：\n',classification_report(test_Y,y_hat))
    # print("KNN均方差为：\n", mean_squared_error(test_Y, y_hat))
    # print('-' * 20)
    acc=accuracy_score(test_Y,y_hat)  #精确度
    cr=classification_report(test_Y,y_hat).replace('\n',' ').split()  #混淆矩阵
    cr=list(map(float,[cr[5],cr[6],cr[7],cr[8],cr[10],cr[11],cr[12],cr[13],cr[15],cr[16],cr[19],cr[20],cr[21],cr[22],cr[25],cr[26],cr[27],cr[28],]))
    matrix=confusion_matrix(test_Y, y_hat)  #KNN分类报告
    mse=mean_squared_error(test_Y, y_hat)   #KNN均方差
    return acc,matrix,cr,mse,knn

#随机深林
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
def RF(train_X=X_train, train_Y=y_train, test_X=X_test, test_Y=y_test):
    # print("Random Forest Classifier")
    # t1 = time.time()
    rfc = RandomForestClassifier()
    rfc.fit(train_X, train_Y)
    y_hat = rfc.predict(test_X)
    # t2 = time.time()
    # print('精确度:', accuracy_score(test_Y, y_hat)) 
    # print('混淆矩阵：\n',confusion_matrix(test_Y, y_hat))
    # print("随机森林-均方差为：\n", mean_squared_error(test_Y, y_hat))
    # print('随即森林分类报告：\n',classification_report(test_Y, y_hat))
    # print('-' * 40)
    # fig = plt.figure(figsize=(20,10))
    # feature_name=data.drop(['Label'],axis=1).columns
    # # ax1 = fig.add_subplot(121)
    # skplt.estimators.plot_feature_importances(rfc,feature_names=feature_name,
    #                                         title="Random Forest Regressor Feature Importance",
    #                                         x_tick_rotation=90)
    # print(rfc.feature_importances_)
    acc=accuracy_score(test_Y,y_hat)
    cr=classification_report(test_Y,y_hat).replace('\n',' ').split()
    cr=list(map(float,[cr[5],cr[6],cr[7],cr[8],cr[10],cr[11],cr[12],cr[13],cr[15],cr[16],cr[19],cr[20],cr[21],cr[22],cr[25],cr[26],cr[27],cr[28],]))
    matrix=confusion_matrix(test_Y, y_hat)
    mse=mean_squared_error(test_Y, y_hat)
    return acc,matrix,cr,mse,rfc
    # print('------------\n')
    # # plt.tight_layout()
    # plt.show()
    # plot_confusion_matrix(matrix, label, True, 'RF Confusion matrix')


from sklearn.svm import SVC
def SVM(train_X=X_train, train_Y=y_train, test_X=X_test, test_Y=y_test):
    # print('[SVM] train ...')
    # train_Y = [np.where(r == 1)[0][0] for r in train_Y]
    # test_Y = [np.where(r == 1)[0][0] for r in test_Y]
    # t1 = time.time()
    clf = SVC(decision_function_shape='ovr', max_iter=300, kernel='rbf')
    model = clf.fit(train_X, train_Y)
    y_hat = model.predict(test_X)
    # t2 = time.time()
    # print('精确度:', accuracy_score(test_Y, y_hat))
    # print('耗时:', t2 - t1, '秒')    
    # print('混淆矩阵：\n',confusion_matrix(test_Y, y_hat))
    # print("SVM均方差为：\n", mean_squared_error(test_Y, y_hat))
    # print('SVM分类报告：\n',classification_report(test_Y, y_hat))
    fig = plt.figure(figsize=(50,15))
    
    # ax1 = fig.add_subplot(122)
    # skplt.estimators.plot_feature_importances(model,feature_names=feature_name,
    #                                         title="Random Forest Regressor Feature Importance",
    #                                         x_tick_rotation=90, order="ascending")
    # plt.tight_layout()
    # plt.show()
    acc=accuracy_score(test_Y,y_hat)
    cr=classification_report(test_Y,y_hat).replace('\n',' ').split()
    cr=list(map(float,[cr[5],cr[6],cr[7],cr[8],cr[10],cr[11],cr[12],cr[13],cr[15],cr[16],cr[19],cr[20],cr[21],cr[22],cr[25],cr[26],cr[27],cr[28],]))
    matrix=confusion_matrix(test_Y, y_hat)
    mse=mean_squared_error(test_Y, y_hat)
    return acc,matrix,cr,mse,clf
    # plot_confusion_matrix(matrix, label, True, 'SVM Confusion matrix')

from sklearn.naive_bayes import BernoulliNB
def NaiveBayes(train_X=X_train, train_Y=y_train, test_X=X_test, test_Y=y_test):
    # print('[Naive Bayes] train ...')
    # train_Y = [np.where(r == 1)[0][0] for r in train_Y]
    # test_Y = [np.where(r == 1)[0][0] for r in test_Y]
    # t1 = time.time()
    clf = BernoulliNB()
    model = clf.fit(train_X, train_Y)
    y_hat = model.predict(test_X)
    acc = accuracy_score(test_Y, y_hat)
    acc=accuracy_score(test_Y,y_hat)
    cr=classification_report(test_Y,y_hat).replace('\n',' ').split()
    cr=list(map(float,[cr[5],cr[6],cr[7],cr[8],cr[10],cr[11],cr[12],cr[13],cr[15],cr[16],cr[19],cr[20],cr[21],cr[22],cr[25],cr[26],cr[27],cr[28],]))
    matrix=confusion_matrix(test_Y, y_hat)
    mse=mean_squared_error(test_Y, y_hat)
    return acc,matrix,cr,mse,clf
    # t2 = time.time()
    # print('精确度:', accuracy_score(test_Y, y_hat))
    # print('耗时:', t2 - t1, '秒')    
    # print('混淆矩阵：\n',confusion_matrix(test_Y, y_hat))
    # print("朴素贝叶斯均方差为：\n", mean_squared_error(test_Y, y_hat))
    # print('朴素贝叶斯分类报告：\n',classification_report(test_Y, y_hat))
    # print('-' * 40)
    # plot_confusion_matrix(matrix, label, True, 'NB Confusion matrix')


#决策树
from sklearn.tree import DecisionTreeClassifier,plot_tree
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def tree3(clf):
    fig = plt.figure(figsize=(35, 10))
    plot_tree(clf, fontsize=8)
    fig.savefig(os.path.join('./', "tree3.png"))


def DT(train_X=X_train, train_Y=y_train, test_X=X_test, test_Y=y_test):
    # t1 = time.time()
    clf = DecisionTreeClassifier(max_depth=5)
    model = clf.fit(train_X, train_Y)
    # print(clf.feature_importances_)
    y_hat = model.predict(test_X)
    acc=accuracy_score(test_Y,y_hat)
    cr=classification_report(test_Y,y_hat).replace('\n',' ').split()
    cr=list(map(float,[cr[5],cr[6],cr[7],cr[8],cr[10],cr[11],cr[12],cr[13],cr[15],cr[16],cr[19],cr[20],cr[21],cr[22],cr[25],cr[26],cr[27],cr[28],]))
    matrix=confusion_matrix(test_Y, y_hat)
    mse=mean_squared_error(test_Y, y_hat)
    return acc,matrix,cr,mse,clf
    # t2 = time.time()
    # print('精确度:', accuracy_score(test_Y, y_hat))
    # print('耗时:', t2 - t1, '秒')    
    # print('混淆矩阵：\n',confusion_matrix(test_Y, y_hat))
    # print("决策树均方差为：\n", mean_squared_error(test_Y, y_hat))
    # print('决策树分类报告：\n',classification_report(test_Y, y_hat))
    # fig = plt.figure(figsize=(50,15))
    # skplt.estimators.plot_feature_importances(model,feature_names=feature_name,
    #                                         title="Random Forest Regressor Feature Importance",
    #                                         x_tick_rotation=90)
    # plt.tight_layout()
    # plt.show()
    # print('-' * 40)
    # plot_confusion_matrix(matrix, label, True, 'DT Confusion matrix')

def return_info():
    if len(re)==0:
        knn_result=KNN()
        rf_result=RF()
        svm_result=SVM()
        nb_result=NaiveBayes()
        dt_result=DT()
        acc={'knn':knn_result[0],'rf':rf_result[0],'svm':svm_result[0],'nb':nb_result[0],'dt':dt_result[0]}
        matrix={'knn':knn_result[1],'rf':rf_result[1],'svm':svm_result[1],'nb':nb_result[1],'dt':dt_result[1]}
        cr={'knn':knn_result[2],'rf':rf_result[2],'svm':svm_result[2],'nb':nb_result[2],'dt':dt_result[2]}
        mse={'knn_':knn_result[3],'rf':rf_result[3],'svm':svm_result[3],'nb':nb_result[3],'dt':dt_result[3]}
        clf={'knn':knn_result[4],'rf':rf_result[4],'svm':svm_result[4],'nb':nb_result[4],'dt':dt_result[4]}
        re.append(acc)
        re.append(matrix)
        re.append(cr)
        re.append(mse)
        re.append(clf)
    return  re
re = return_info()
print(type(re[2]['knn']))

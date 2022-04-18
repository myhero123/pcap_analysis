from __future__ import division
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm, metrics
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix
import seaborn as sns
import threading
from sklearn.datasets.samples_generator import make_classification
from sklearn.neighbors import KNeighborsClassifier
from bubbly.bubbly import bubbleplot
from plotly.offline import init_notebook_mode, iplot
from sanwei import k_means_run


def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.winter):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] < thresh else "black", fontsize=40)

    plt.tight_layout()
    plt.ylabel('True label', fontsize=30)
    plt.xlabel('Predicted label', fontsize=30)
    return plt


def JZ(juzheng):
    #confusion = np.array(([91, 0, 0,4], [0, 92, 1,4], [0, 0, 95,4],[1,1,1,1]))
    confusion = juzheng
    # 热度图，后面是指定的颜色块，可设置其他的不同颜色
    plt.imshow(confusion, cmap=plt.cm.Blues)
    # ticks 坐标轴的坐标点
    # label 坐标轴标签说明
    indices = range(len(confusion))
    plt.xticks(indices, [1, 3, 3,4,5,6,7,8,9,10])
    plt.yticks(indices, [1, 2, 3,4,5,6,7,8,9,10])
    plt.colorbar()
    plt.xlabel('预测值')
    plt.ylabel('真实值')
    plt.title('混淆矩阵')
    sns.set(font_scale=0.8)
    # plt.rcParams两行是用于解决标签不能显示汉字的问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 显示数据
    for first_index in range(len(confusion)):  # 第几行
        for second_index in range(len(confusion[first_index])):  # 第几列
            plt.text(first_index, second_index, confusion[first_index][second_index])
    # 在matlab里面可以对矩阵直接imagesc(confusion)
    # 显示
    plt.show()
def shan_dian(y_test,y_predict):
    plt.scatter(y_test, y_predict)
    plt.show()

def xiao_ti_qing(a):
    #小提琴图
    a['categoryname'].value_counts()  # 对分类变量的类别进行计数
    # 后面将研究不同类型的'gearbox'对应'price'的差异
    x = a['categoryname']
    y = a['notified']  # 在原数据集中，'price'为目标变量
    # 绘制小提琴图
    sns.violinplot(x=x, y=y, data=a)
    plt.show()
    # 在sns.violinplot中，x是类别变量，y是数值型变量，data用于指定数据集

def xiang_guan(a):
    # 特征间相关系数热力图
    f = a.corr()
    sns.heatmap(f, annot=True)
    plt.show()
def categoryname(a):
    X = a.drop(["categoryname"], axis=1)  # 11829
    y = a["categoryname"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # 标准化
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Random Forest Classifier
    print("***************[categoryname]*****************")
    print("Random Forest Classifier")
    # n_estimators，表示选择多少棵树来构造随机森林；具体解释看《边学边练超系统掌握人工智机器学习算法 传智播客》
    rfc = RandomForestClassifier(n_estimators=200)
    rfc.fit(X_train, y_train)
    pred_rfc = rfc.predict(X_test)
    print("Random Forest classification_report==\n", classification_report(y_test, pred_rfc))
    print("Random Forest confusion_matrix==\n", confusion_matrix(y_test, pred_rfc))
    # 混淆矩阵可视化
    # t1 = threading.Thread(target=JZ,args=[confusion_matrix(y_test, pred_rfc)])
    # t1.start()
    JZ(confusion_matrix(y_test, pred_rfc))
def notified(a):
    X = a.drop(["notified"], axis=1)  # 11829
    y = a["notified"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # 标准化
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Random Forest Classifier
    print("******************[notified]******************")
    print("Random Forest Classifier")
    # n_estimators，表示选择多少棵树来构造随机森林；具体解释看《边学边练超系统掌握人工智机器学习算法 传智播客》
    rfc = RandomForestClassifier(n_estimators=200)
    rfc.fit(X_train, y_train)
    pred_rfc = rfc.predict(X_test)
    print("Random Forest classification_report==\n", classification_report(y_test, pred_rfc))
    print("Random Forest confusion_matrix==\n", confusion_matrix(y_test, pred_rfc))
    error = mean_squared_error(y_test, pred_rfc)
    print("随机森林-均方差为：\n", error)

    estimator = KNeighborsClassifier(n_neighbors=3)
    estimator.fit(X_train, y_train)
    print("KNN classification_report==\n", classification_report(y_test, estimator.predict(X_test)))
    error = mean_squared_error(y_test, estimator.predict(X_test))
    print("KNN-均方差为：\n", error)


    # SVM Classifier
    print("SVM Classifier")
    svmc = svm.SVC()
    svmc.fit(X_train, y_train)
    pred_svmc = svmc.predict(X_test)
    print("SVM classification_report==\n", classification_report(y_test, pred_svmc))
    print("SVM classification_confusion_matrix==\n", confusion_matrix(y_test, pred_svmc))
    error = mean_squared_error(y_test, pred_svmc)
    print("SVM-均方误差为：\n", error)

    # Neural Network
    print("Neural Network")
    mlpc = MLPClassifier(hidden_layer_sizes=(11, 11, 11), max_iter=500)
    mlpc.fit(X_train, y_train)
    pred_mlpc = mlpc.predict(X_test)
    print("Neural Network classification_report==\n", classification_report(y_test, pred_mlpc))
    print("Neural Network confusion_matrix==\n", confusion_matrix(y_test, pred_mlpc))
    error = mean_squared_error(y_test, pred_mlpc)
    print("神经网络-均方误差为：\n", error)

def main():
    path = "cybersecurity_training.csv"
    a = pd.read_csv(path, delimiter=",")
    # drop（）方法是
    a = a.drop(columns=['alert_ids', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'score',
                        'grandparent_category', 'weekday', 'ip', 'ipcategory_name', 'ipcategory_scope',
                        'parent_category',
                        'timestamp_dist', 'start_hour', 'start_minute', 'start_second', 'thrcnt_month', 'thrcnt_week',
                        'thrcnt_day'])

    col_str = ['client_code', 'categoryname', 'dstipcategory_dominate', 'srcipcategory_dominate']
    print(len(a))
    # Converting non-numeric data into numeric data
    ## 离散型的数据转换成 0 0 0 到 n − 1 n-1 n−1 之间的数
    for i in col_str:
        label_enc = LabelEncoder()
        label_enc.fit(a[i])
        a[i] = label_enc.transform(a[i])

    # 特征间相关系数热力图
    a_ = a[['notified','categoryname','dstipcategory_dominate','srcipcategory_dominate','dstport_cd','alerttype_cd',\
            'eventname_cd','severity_cd','reportingdevice_cd','devicetype_cd','devicevendor_cd','domain_cd','protocol_cd','username_cd',\
            'srcipcategory_cd','dstipcategory_cd','isiptrusted','untrustscore','flowscore','trustscore','enforcementscore',\
            'dstipcategory_dominate','srcipcategory_dominate','dstportcategory_dominate','srcportcategory_dominate']]
    xiang_guan(a_)

    #受到攻击是否收到警告通知
    notified(a)

    # categoryname混淆矩阵可视化
    # t1 = threading.Thread(target=JZ,args=[confusion_matrix(y_test, pred_rfc)])
    # t1.start()
    categoryname(a)

    # 哪几种类别攻击的严重程度更容易被检测到
    xiao_ti_qing(a)

    #k_means聚类三维可视化
    k_means_run()
   


if __name__ == '__main__':
    main()
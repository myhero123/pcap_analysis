import machine
import feature_engineering as fe
X_train, X_test, y_train, y_test=fe.Data_preprocessing()

# a=machine.DT(X_train,y_train,X_test,y_test)
SVM_re=machine.DTSVM(X_train,y_train,X_test,y_test)
print(SVM_re)
import pandas as pd
import numpy as np
from sklearn import neighbors
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold

data = pd.read_csv("dat.csv")

print(data)
data_Y = data["Risk"];


lb_month = LabelEncoder()
data["month_code"] = lb_month.fit_transform(data["Month"])
data[["Month", "month_code"]].head()
data.head(57)


lb_year = LabelEncoder()
data["year_code"] = lb_month.fit_transform(data["Year"])
data[["Year", "year_code"]].head()
data.head(57)


lb_month = LabelEncoder()
data["road_code"] = lb_month.fit_transform(data["Roads"])
data[["Roads", "road_code"]].head()
data.head(57)



print(data);


data=data.drop("Month", 1)
data=data.drop("Roads", 1)
data=data.drop("Risk", 1)
data=data.drop("Year", 1)

print(data);
print(data_Y);

x = np.array(data)
y = np.array(data_Y)

k = KFold(n_splits=9,random_state=33,shuffle = True)
#k = KFold(n_splits = 7,random_state = 0,shuffle = True)

for train_index,test_index in k.split(x):
    
    x_train, x_test= x[train_index], x[test_index]
    y_train, y_test= y[train_index], y[test_index]
    
    #x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
    

    scaler = preprocessing.StandardScaler().fit(x_train)
    
    
knn=neighbors.KNeighborsClassifier(3)    
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
knn.fit(x_train,y_train)
y_train_pred = knn.predict(x_train)
print(y_train_pred);
print ("\nAccuracy of training: ",metrics.accuracy_score(y_train, y_train_pred),"\n")
y_pred = knn.predict(x_test)
print ("Accuracy of testing: ",metrics.accuracy_score(y_test, y_pred),"\n")
print(metrics.classification_report(y_test, y_pred,target_names=data_Y),"\n")
            
        
svma = svm.SVC()
svma.fit(x_train, y_train)
y_train_pred = svma.predict(x_train)
print(y_train_pred);
print ("\nAccuracy of training: ",metrics.accuracy_score(y_train, y_train_pred),"\n")
y_pred = svma.predict(x_test)
print ("Accuracy of testing: ",metrics.accuracy_score(y_test, y_pred),"\n")
print(metrics.classification_report(y_test, y_pred,target_names=data_Y),"\n")


    
SGD =  SGDClassifier()
SGD.fit(x_train,y_train)
y_train_pred = SGD.predict(x_train)
print(y_train_pred);
print ("\nAccuracy of training: ",metrics.accuracy_score(y_train, y_train_pred),"\n")
y_pred = SGD.predict(x_test)
print ("Accuracy of testing: ",metrics.accuracy_score(y_test, y_pred),"\n")
print(metrics.classification_report(y_test, y_pred,target_names=data_Y),"\n")

     
ad = AdaBoostClassifier()
ad.fit(x_train,y_train)
y_train_pred = ad.predict(x_train)
print(y_train_pred);
print ("\nAccuracy of training: ",metrics.accuracy_score(y_train, y_train_pred),"\n")
y_pred = ad.predict(x_test)
print ("Accuracy of testing: ",metrics.accuracy_score(y_test, y_pred),"\n")
print(metrics.classification_report(y_test, y_pred,target_names=data_Y),"\n")


rf = RandomForestClassifier()
rf.fit(x_train,y_train)
y_train_pred = rf.predict(x_train)
print(y_train_pred);
print ("\nAccuracy of training: ",metrics.accuracy_score(y_train, y_train_pred),"\n")
y_pred = rf.predict(x_test)
print ("\nAccuracy of testing: ",metrics.accuracy_score(y_test, y_pred),"\n")
print(metrics.classification_report(y_test, y_pred,target_names=data_Y),"\n")

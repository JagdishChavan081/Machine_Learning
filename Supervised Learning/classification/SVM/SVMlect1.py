#svm-support vector machine, classification algorithm, small data set ,
# applied in OCR - optical character recognitation, churn data in bank ect
#importing required libreries
import pandas as pd
#for step3 for training and testing purpose
from sklearn.model_selection import train_test_split
#for step 3.1 standard scalling of data
from sklearn.preprocessing import StandardScaler
#for state 3.2 calling classifier
from sklearn.svm import SVC
#for checking accuracy and conclusion
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#step1 reading data
data = pd.read_csv('/home/jagdish/pytn/ml/Dset/Social_Network_Ads.csv')
print(data.head())

#step2 dividing the data set into independent and dependent variable
data_x = data.iloc[:,[2,3]].values
data_y = data.iloc[:,4].values

#step3 spliting dataset into training and testing variable
x_train,x_test,y_train,y_test =train_test_split(data_x,data_y,
                                                random_state=0, test_size=0.25)


#step3.1 standard scalling of data
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test =sc.fit_transform(x_test)

#step3.2 classifing data
cls = SVC(random_state= 0)
cls.fit(x_train,y_train)

y_pred = cls.predict(x_test)
print(y_pred)
print(y_test)


#step4
con = confusion_matrix(y_test,y_pred)
print(con)

acc = accuracy_score(y_test,y_pred)
print(acc)









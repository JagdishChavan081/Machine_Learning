#its name is logistic regression but performs classification
#step 1 importing necessary libraries
import pandas as pd
#step2 importing for performing training and testing
from sklearn.model_selection import train_test_split
#step3.1for performimg standard scaling
from sklearn.preprocessing import StandardScaler
#step4 calling Algorithm to be used
from sklearn.linear_model import LogisticRegression
#step4.1 comparing o/p using confusion matrix
from sklearn.metrics import confusion_matrix
#for checking accuracy score
from sklearn.metrics import accuracy_score


#step1 importing and reading data set
data = pd.read_csv('/home/jagdish/pytn/ml/Dset/Social_Network_Ads.csv')
print(data.head())

#step2 split data into dependent and independent variable
data_x = data.iloc[:,[2,3]].values
data_y = data.iloc[:,4].values

#Step3 spliting data according to tarining and spliting
x_train,x_test,y_train,y_test =train_test_split(data_x,data_y,
                                random_state=0, test_size=0.25)


#step3.1 standard scalling
sc = StandardScaler()
x_train =sc.fit_transform(x_train)
x_test = sc.transform(x_test)
#print(x_test)

#step4 building classifier using algorithm and training
cls = LogisticRegression(random_state=0)
cls.fit(x_train,y_train)

y_pred =cls.predict(x_test)
print(y_pred)
print(y_test)

#comparing output
cm = confusion_matrix(y_test,y_pred)
print(cm)

#cheacking accuracy of model
ac =  accuracy_score(y_test,y_pred)
print(ac)







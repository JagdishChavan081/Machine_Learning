import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


data =pd.read_csv('/home/jagdish/pytn/ml/Dset/stu.csv')
print(data.head())
le = preprocessing.LabelEncoder()
data['subject']=le.fit_transform(data['subject'])
data['pass/fail']=le.fit_transform(data['pass/fail'])
print(data)

#step2 splitting into Independent & dependent variable
#getting all the value other than last column which is no of rides driver has to carry on
data_x = data.iloc[:,:-1].values
#print(data_x)
#getting all the value from last column
data_y = data.iloc[:,-1].values
#print(data_y)

#step3 training and testing set
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3,random_state=0)
print("x_tarin,y_tarin",x_train,y_train)
print("x_test, y_test",x_test,y_test)

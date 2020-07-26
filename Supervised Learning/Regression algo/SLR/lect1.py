#importing Libreries
import pandas as pd
#for step three onwards
from sklearn.model_selection import train_test_split
#for step 3.1
from sklearn.linear_model import LinearRegression
#For visualization we use
import matplotlib.pyplot as plt

#step 1 read Data
data = pd.read_csv('/Dset/taxi.csv')
#print(data.head(8))

#step2 splitting into Independent & dependent variable
#getting all the value other than last column which is no of rides driver has to carry on
data_x = data.iloc[:,:-1].values
#print(data_x)
#getting all the value from last column
data_y = data.iloc[:,-1].values
#print(data_y)

#step3 training and testing set
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3,random_state=0)

#step3.1applying algorithm
reg = LinearRegression()  #y=mx+c

#step4 Model training
reg.fit(x_train,y_train)

#step5 prediction
y_pred = reg.predict(x_test)
print("predicted value:-",y_pred)

#comparing with real data
print("\nactual value",y_test)

#visualizing data
# plt.scatter(x_train, y_train, color ='red')
# plt.title("Exp Vs Salary")
# plt.xlabel("Exp.")
# plt.ylabel("salary")
# plt.show()















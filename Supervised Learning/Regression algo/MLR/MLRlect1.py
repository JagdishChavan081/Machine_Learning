#importing Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


#reading data set
data = pd.read_csv('/home/jagdish/pytn/ml/Dset/taxi.csv')

#printing Data set
print(data.head())

#seprating input and output variable
#y=b0 +b1x1 +b2x2 +b3x3
#sepreting x1,x2,x3
data_x=data.iloc[:,0:-1].values
print(data_x)

#sepreting y
data_y=data.iloc[:,-1].values
print(data_y)

x_train, x_test, y_train, y_test=train_test_split(data_x, data_y,
                            random_state=0, test_size=0.3)


#training the model
reg =LinearRegression()
reg.fit(x_train,y_train)


y_pred = reg.predict(x_test)
print("\n prediction output:-",y_pred)

#comparing with original value
print("\n original data:-",y_test)

#getting value of cofficient(b1,b2,b3)
print("\n coffecient b:-",reg.coef_)

#getting value of intercept b0,c
print("\n intercept b0:-",reg.intercept_)

#equation of multiple linear equation for this data set
#y = b0 + b1x1 + b2x2 +b3x3 +b4x4
#y=number of weekly rides
#x1 = priceperweek
#x2 =population
#x3 =Monthly Income
#x4 = average parking per month
y0 = (reg.intercept_ + reg.coef_[0]*x_test[0][0]
                    + reg.coef_[1]*x_test[0][1]
                    +reg.coef_[2]*x_test[0][2]
                    +reg.coef_[3]*x_test[0][3])
print("\n value of y0:-",y0)

#checking score of regression
print('\nTraining Score:-',reg.score(x_train,y_train))
print('\nTesting Score:-',reg.score(x_test,y_test))

#checking for random value
X1 = [[80, 1770000, 6000, 85]]
output = reg.predict(X1)
print("number of weekly riders:-",output)


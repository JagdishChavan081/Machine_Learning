
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt



data = pd.read_csv('/home/jagdish/pytn/ml/Dset/salary.csv')
#print(data.head())


data_x = data.iloc[:,:-1].values
data_y = data.iloc[:,-1].values

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3,random_state=42)

reg = LinearRegression()


reg.fit(x_train,y_train)

y_pred = reg.predict([[3.8]])
#print("predicted value:-",y_pred)

#print("\nactual value",y_test)

plt.scatter(x_train, y_train, color ='red')
plt.plot(x_train,LinearRegression.predict(x_train),color='blue')
plt.title("Exp Vs Salary trainning")
plt.xlabel("Exp.")
plt.ylabel("salary")
plt.show()

# plt.scatter(x_test, y_test, color ='red')
# plt.plot_date(x_train,LinearRegression.predict(x_train),color='blue')
# plt.title("Exp Vs Salary testing ")
# plt.xlabel("Exp.")
# plt.ylabel("salary")
# plt.show()
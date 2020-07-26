#importing necessary libraries
import pandas as pd
#for step 2
from sklearn.model_selection import train_test_split
#for feature scalling
from sklearn.preprocessing import StandardScaler
#for using KNN algorithm
from sklearn.neighbors import KNeighborsClassifier
#for confusion matrix
from sklearn.metrics import confusion_matrix
#for checking accuracy of confusion matrics
from sklearn.metrics import accuracy_score




#step1 calling and reading data
data = pd.read_csv('/home/jagdish/pytn/ml/Dset/Social_Network_Ads.csv')
print(data.head())
#400 obsrvation point i.e 400 rows
#print(len(data))
#looking at the data we can see for our operation user id is not necessary gender dosent matter
# in trems of product like  mobile phone hence
#independent or featured variable are Age and EstimatedSalary
#dependent variable is purchased

#step2 spliting the data
data_x = data.iloc[:,[2,3]].values
data_y =data.iloc[:,-1].values

#Step3 spliting data as per training and testing
x_train,x_test,y_train,y_test =train_test_split(data_x,data_y, random_state=0, test_size=0.25)

#feature scaling
#looking into data set we can see that value for age and sallery are having to much difference
#which becomes nearly impossible for graph plotting
#hence we perform feature scalling which devides paremeter between +2 to -2
sc = StandardScaler()
x_train =sc.fit_transform(x_train)
x_test = sc.transform(x_test)
#print(x_test)

#step4 training Applying KNN for tarining
#p=1 manhatan distance , p=2 euclidian distance
cls = KNeighborsClassifier(n_neighbors=5, metric='minkowski',p=2)
cls.fit(x_train,y_train)

#classification
y_pred = cls.predict(x_test)
print(y_pred)
#0 = not intrested in buying 1= potential customer

print(y_test)

#comparision using confusion matrix
cm = confusion_matrix(y_test,y_pred)
print("\n cm",cm)

#for checking accuracy of cm
ac = accuracy_score(y_test,y_pred)
print('\n ac',ac)







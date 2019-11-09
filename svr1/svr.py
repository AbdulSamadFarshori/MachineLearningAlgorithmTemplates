#import libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

#import dataset
df = pd.read_csv('salaries.csv',skipinitialspace = True)

#slice data into x and y
x = df.iloc[:,1].values
y = df.iloc[:,2].values

#create pipeline

#handle missing values
'''from sklearn.impute import SimpleImputer
im = SimpleImputer(missing_values=np.nan,strategy = 'median', axis =0 )'''

#features scaling 
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_y.fit_transform(x.reshape((10,1)))
y = sc_x.fit_transform(y.reshape((10,1)))
#split data
'''from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)'''

#strat making svr model
from sklearn.svm import SVR
regressor =  SVR(kernel='rbf')
regressor.fit(x.reshape((10,1)),y.reshape((10,)))
#predict our model
predict =sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array([[6.5]]))))
#visualisation 
plt.scatter(x,y, color='red')
plt.plot(x,regressor.predict(x),color='green')
plt.show()
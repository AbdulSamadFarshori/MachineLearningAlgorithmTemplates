# import libaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import data 
df = pd.read_csv('salaries.csv', skipinitialspace = True)

#seperate features and label
x = df.iloc[:,1].values
y = df.iloc[:,2].values

#create pipeline 

# handle missing values

'''from sklearn.impute import SimpleImputer 
im = SimpleImputer(missing_values = np.nan, strategy = 'median', axis = 0)'''

# features scaling

'''from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)
y = sc.fit_transform(y)'''

#split dataset into training and test set
'''from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)'''


#start making random forest regression model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 250, random_state = 0)
regressor.fit(x.reshape((10,1)),y)

#predict our model

predict = regressor.predict(x.reshape(10,1))

#visualization
x_grid = np.arange(min(x),max(x),0.01)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x,y, color = 'red')
plt.plot(x_grid,regressor.predict(x_grid), color = 'blue')
plt.show()
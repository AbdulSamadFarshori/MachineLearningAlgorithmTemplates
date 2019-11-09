import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# import dataset
df = pd.read_csv('salaries.csv')

#slice data into x and y 
x = df.iloc[:,1:2].values
y = df.iloc[:,2].values

### Creating Pipeline 

# for missing values

'''from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy='median')
imputer = imputer.fit()'''

# for variables convert into standard unit

'''from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_trasform(x)
y = sc.fit_transform(y)'''


# split data into training and test set

'''from sklearn.model_slection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)'''



# start making poly regression model
from sklearn.preprocessing import PolynomialFeatures
poly_regressor = PolynomialFeatures(degree=4)
x_poly = poly_regressor.fit_transform(x)
poly_regressor.fit(x_poly,y)
# import linear regressor
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_poly,y)


#predict our model
predict = lr.predict(poly_regressor.fit_transform(x))
predict

#draw graph or visualization
x_grid = np.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x,y, color = 'red')
plt.plot(x_grid, lr.predict(poly_regressor.fit_transform(x_grid)),color='blue')
plt.title('salary of position')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()
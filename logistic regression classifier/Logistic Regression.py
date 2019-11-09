#import libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

#import dataset
df = pd.read_csv('Social_Network_Ads.csv',skipinitialspace = True)

#sperate features and label
x = df.iloc[:,[2,3]].values
y = df.iloc[:,4].values

#create pipeline
#handle missing values
'''from sklearn.impute from SimpleImputer
im = SimpleImputer(missing_values= np.nan, strategy = 'median', axis = 0)'''

#split data into training and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

#feature scaling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

#start making logistic regression model
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state = 0)
clf.fit(x_train,y_train)

#test our model
predict = clf.predict(x_test)

#create confusion matarix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,predict)

#visualisation

from matplotlib.colors import ListedColormap
x_set,y_set = x_train,y_train
x1,x2 = np.meshgrid(np.arange(start = x_set[:,0].min()-1,stop = x_set[:,0].max()+1,step =0.03),
                    np.arange(start = x_set[:,1].min()-1,stop = x_set[:,1].max()+1,step= 0.03))
plt.contourf(x1,x2,clf.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
            alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0], x_set[y_set==j,1], c = ListedColormap(('red','green'))(i),label = j)
plt.show()


from matplotlib.colors import ListedColormap
x_set,y_set = x_test,y_test
x1,x2 = np.meshgrid(np.arange(start = x_set[:,0].min()-1,stop = x_set[:,0].max()+1,step =0.01),
                    np.arange(start = x_set[:,1].min()-1,stop = x_set[:,1].max()+1,step= 0.01))
plt.contourf(x1,x2,clf.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
            alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0], x_set[y_set==j,1], c = ListedColormap(('red','green'))(i),label = j)
plt.show()



# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import dataset 
df = pd.read_csv('Social_Network_Ads.csv')

# seperate feature and label
x = df.iloc[:,[2,3]]
y = df.iloc[:,4]

# handle missing values
'''from sklearn.impute import SimpleImputer
im = SimpleImputer(missing_values = np.nan, strategy = 'median', axis = 0)'''

# split dataset into training and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.2, random_state = 42)

# feature scaling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

# start making decision tree classifier
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion ='entropy',random_state = 0)
clf.fit(x_train,y_train)

# predict our model
predict = clf.predict(x_test)

#create confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,predict)
#visualization of train data
from matplotlib.colors import ListedColormap
x_set,y_set = x_train,y_train
x1,x2 = np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop= x_set[:,0].max()+1,step=0.01),
                    np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
plt.contourf(x1,x2,clf.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
             alpha =0.75, cmap = ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set ==j,0],x_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)
plt.show()

#visualization of test data
from matplotlib.colors import ListedColormap
x_set,y_set = x_test,y_test
x1,x2 = np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop= x_set[:,0].max()+1,step=0.01),
                    np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
plt.contourf(x1,x2,clf.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
             alpha =0.75, cmap = ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set ==j,0],x_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)
plt.show()



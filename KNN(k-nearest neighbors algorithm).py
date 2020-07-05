
#importing necessary packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

#reading the CSV datafile.

df=pd.read_csv("/content/Student.csv")
df.head()   #prints the first 10rows

#droping NULL value

df.dropna(inplace=True)

#FeatureSelection

X=df[['Physics','Chemistry','Maths']]
y=df['Result']

#modelcreation

clf=KNeighborsClassifier(n_neighbors=5,metric='euclidean')

#fitting model

clf.fit(X,y)

#making prediction

print(clf.predict([(90,90,0)]))

array([1])   #output


____________________________________________________________________________________________________________________________________________________________________________________________







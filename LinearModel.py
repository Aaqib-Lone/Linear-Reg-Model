# First we need to import the modules/Librarires
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

# now read the csv file with pandas 
df=pd.read_csv("student-mat.csv", sep=";")
df=df[["G1","G2", "G3", "studytime", "failures", "absences"]]

predict="G3"

# Assign the labels with the help of numpy array
X=np.array(df.drop([predict],1))
y=np.array(df[predict])

# Now it's the time to train our model with
X_train,  x_test,y_train, y_test= train_test_split(X,y, test_size=0.1)

# Apply the linear regression algorithm
reg=linear_model.LinearRegression()
reg.fit(X_train, y_train)

# find the accuracy
accuracy=reg.score(x_test, y_test)

print("Accuracy : ",accuracy)

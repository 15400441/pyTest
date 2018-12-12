import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

#load train data
X_train = pd.read_csv('X_train.csv')
Y_train = pd.read_csv('Y_train.csv')

#load test data
X_test = pd.read_csv('X_test.csv')
Y_test = pd.read_csv('Y_test.csv')

#encodeLabels
le=LabelEncoder()
for col in X_test.columns.values:
    # Encoding only categorical variables
    if X_test[col].dtypes=='object':
        # Using whole data to form an exhaustive list of levels
        data=X_train[col].append(X_test[col])
        le.fit(data.values)
        X_train[col]=le.transform(X_train[col])
        X_test[col]=le.transform(X_test[col])

X_train_scale=scale(X_train)
X_test_scale=scale(X_test)
log=LogisticRegression(penalty='l2',C=.01)
log.fit(X_train_scale,Y_train)
score =  accuracy_score(Y_test,log.predict(X_test_scale))
print(score)

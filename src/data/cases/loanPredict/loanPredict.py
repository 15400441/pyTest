import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

#load train data
X_train = pd.read_csv('X_train.csv')
Y_train = pd.read_csv('Y_train.csv')

#load test data
X_test = pd.read_csv('X_test.csv')
Y_test = pd.read_csv('Y_test.csv')

#visualize feature data
#X_train[X_train.dtypes[(X_train.dtypes=="float64")|(X_train.dtypes=="int64")].index.values].hist(figsize=[11,11])
#plt.show()

##########################################  knn  #####################################################
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train[['ApplicantIncome', 'CoapplicantIncome','LoanAmount',
                 'Loan_Amount_Term', 'Credit_History']],Y_train)

score = accuracy_score(Y_test,knn.predict(X_test[['ApplicantIncome', 'CoapplicantIncome',
                                          'LoanAmount', 'Loan_Amount_Term', 'Credit_History']]))
print('using knn before scaling, score='+str(score))

# Scaling down both train and test data set
min_max=MinMaxScaler()
X_train_minmax=min_max.fit_transform(X_train[['ApplicantIncome', 'CoapplicantIncome',
                                              'LoanAmount', 'Loan_Amount_Term', 'Credit_History']])
X_test_minmax=min_max.fit_transform(X_test[['ApplicantIncome', 'CoapplicantIncome',
                                            'LoanAmount', 'Loan_Amount_Term', 'Credit_History']])
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_minmax,Y_train)
score = accuracy_score(Y_test,knn.predict(X_test_minmax))
print('using knn after scaling, score='+str(score))

######################################### logisticRegression  #######################################################
log = LogisticRegression(penalty='l2',C=.01)
log.fit(X_train[['ApplicantIncome', 'CoapplicantIncome',
                 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']],Y_train)
score = accuracy_score(Y_test,log.predict(X_test[['ApplicantIncome', 'CoapplicantIncome',
                                          'LoanAmount', 'Loan_Amount_Term', 'Credit_History']]))
print('using logisticRegression before scaling, score='+str(score))


X_train_scale=scale(X_train[['ApplicantIncome', 'CoapplicantIncome',
                             'LoanAmount', 'Loan_Amount_Term', 'Credit_History']])
X_test_scale=scale(X_test[['ApplicantIncome', 'CoapplicantIncome',
                           'LoanAmount', 'Loan_Amount_Term', 'Credit_History']])
log=LogisticRegression(penalty='l2',C=.01)
log.fit(X_train_scale,Y_train)
score = accuracy_score(Y_test,log.predict(X_test_scale))
print('using logisticRegression after scaling, score='+str(score))

#######################################################################################################################

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
clf = DecisionTreeClassifier()
clf.fit(X_train_scale,Y_train)
score = accuracy_score(Y_test,clf.predict(X_test_scale))
print('using decisionTree after scaling, score='+str(score))

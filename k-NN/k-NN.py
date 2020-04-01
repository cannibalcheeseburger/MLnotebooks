import pandas as pd
import numpy as np

df = pd.read_csv("./diabetes.csv")
X = df.drop(['Outcome'],axis = 1)
Y = df['Outcome']


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(X)
#MinmaxScaler.fit() will return a an array which needs to be converted into pandas Dataframe
X = pd.DataFrame(x_scaled,columns = X.columns)


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state = 50,stratify =  Y)


from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import f1_score

def Elbow(k):
    test_f1 = []
    train_f1 = []
    for i in k:
        clf = KNN(n_neighbors=i)
        clf.fit(X_train,Y_train)
        pred = clf.predict(X_test)
        error = f1_score(pred,Y_test)
        test_f1.append(error)
        pred = clf.predict(X_train)
        error = f1_score(pred,Y_train)
        train_f1.append(error)
    return train_f1,test_f1  
          
k  = range(2,50)
f_train,f_test = Elbow(k)

# Creating Knn for least error value
knn = KNN(n_neighbors= 23 )
knn.fit(X_train,Y_train)
Pred = knn.predict(X_test)
F1 = f1_score(Pred,Y_test)
print("F1 score of model is:",F1)

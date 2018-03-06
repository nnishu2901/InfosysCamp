

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import csv
import numpy as ny
import pandas

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


#X = ny.array([[1,2], [3,4], [5,6], [7,8], [9,10]])
#Y = ny.array([3, 4, 5, 6, 7])

X = ny.empty([0,2])
Y = ny.empty([0,1])

with open('Dataset.csv') as csvfile:
    readCSV=csv.reader(csvfile,delimiter=',')
    for row in readCSV:
        X = ny.vstack([X,[row[0],row[1]]])
        Y = ny.vstack([Y,row[2]])
print X
print Y

sc_X=StandardScaler()
X_train = sc_X.fit_transform(X)

Y=ny.reshape(Y,(-1,1))
sc_Y=StandardScaler()
Y_train = sc_Y.fit_transform(Y)


N = 5

def brain():
    #Create the brain
    br_model=Sequential()
    br_model.add(Dense(20, input_dim=2, kernel_initializer='normal',activation='relu'))
    br_model.add(Dense(10, kernel_initializer='normal',activation='relu'))


    br_model.add(Dense(10, kernel_initializer='normal',activation='relu'))
    br_model.add(Dense(10, kernel_initializer='normal',activation='relu'))
  
    br_model.add(Dense(1,kernel_initializer='normal'))
    
    #Compile the brain
    br_model.compile(loss='mean_squared_error',optimizer='adam')
    return br_model

def predict(X,sc_X,sc_Y,estimator):
    prediction = estimator.predict(sc_X.fit_transform(X))
    return sc_Y.inverse_transform(prediction)


estimator = KerasRegressor(build_fn=brain, epochs=10000, batch_size=5,verbose=0)
# print "Done"


#seed = 21
#ny.random.seed(seed)
#kfold = KFold(n_splits=N, random_state=seed)
#results = cross_val_score(estimator, X, Y, cv = kfold)
estimator.fit(X_train,Y_train)
# prediction = estimator.predict(X_train)

# print Y_train
# print prediction


# print Y
#pred_final= sc_Y.inverse_transform(prediction)
pred_final = predict(X,sc_X,sc_Y,estimator)
print pred_final

X_trainOut=ny.empty([0,2])
for i in range(0, len(Y)):
   row_new = [Y[i][0], pred_final[i]]
   X_trainOut=ny.vstack([X_trainOut,row_new])

print X_trainOut

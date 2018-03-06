

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


import numpy as ny
import pandas

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

X = ny.array([[1,2], [3,4], [5,6], [7,8], [9,10]])
sc_X=StandardScaler()
X_train = sc_X.fit_transform(X)

Y = ny.array([3, 4, 5, 6, 7])
Y=ny.reshape(Y,(-1,1))
sc_Y=StandardScaler()
Y_train = sc_Y.fit_transform(Y)

N = 5

def brain():
    #Create the brain
    br_model=Sequential()
    br_model.add(Dense(3, input_dim=2, kernel_initializer='normal',activation='relu'))
    br_model.add(Dense(2, kernel_initializer='normal',activation='relu'))
    br_model.add(Dense(1,kernel_initializer='normal'))
    
    #Compile the brain
    br_model.compile(loss='mean_squared_error',optimizer='adam')
    return br_model


def predict(X,sc_X,sc_Y,estimator):
    prediction = estimator.predict(sc_X.fit_transform(X))
    return sc_Y.inverse_transform(prediction)

estimator = KerasRegressor(build_fn=brain, epochs=1000, batch_size=5,verbose=0)
# print "Done"


#seed = 21
#ny.random.seed(seed)
#kfold = KFold(n_splits=N, random_state=seed)
#results = cross_val_score(estimator, X, Y, cv = kfold)
estimator.fit(X_train,Y_train)
prediction = estimator.predict(X_train)

# print Y_train
# print prediction

print predict(X,sc_X,sc_Y,estimator)

X_test = ny.array([[1.5,4.5], [7,8], [9,10]])
print predict(X_test,sc_X,sc_Y,estimator)

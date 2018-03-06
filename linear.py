import tensorflow as tf
import csv
import numpy as np

from tensorflow.python import debug as tf_debug

#x1_data = [1, 0, 3, 0, 5]
#x2_data = [0, 2, 0, 4, 0]
#y_data = [1, 2, 3, 4, 5]
x1_data=[]
x2_data=[]
y_data=[]
with open('prev_Dataset.csv') as csvfile:
    readCSV=csv.reader(csvfile,delimiter=',')
    for row in readCSV:
        x1_data.extend([float(row[0])])
        x2_data.extend([float(row[1])])
        y_data.extend([float(row[2])])

#print x1_data
#print x2_data
#print y_data

#Normalization starts

min_x1=min(x1_data)
max_x1=max(x1_data)
mean_x1=float(sum(x1_data)/len(x1_data))
dev_x1= max_x1-min_x1

min_x2=min(x2_data)
max_x2=max(x2_data)
mean_x2=float(sum(x2_data)/len(x2_data))
dev_x2=max_x2-min_x2



for i in range(0,len(x1_data)):
    x1_data[i]=(x1_data[i]-mean_x1)/dev_x1
    x2_data[i]=(x2_data[i]-mean_x2)/dev_x2


#Normalization ends


W1 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

hypothesis = W1 * x1_data + W2 * x2_data + b

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in xrange(2001):
    sess.run(train)
    if step % 20 == 0:
        print step, sess.run(cost), sess.run(W1), sess.run(W2), sess.run(b)
        w1=sess.run(W1)
        w2=sess.run(W2)
        B=sess.run(b)

# print w1, w2, B
x1_test=[]
x2_test=[]
y_test=[]
#with open('hypothetical_sentiment.csv') as csvfile:
#    readCSV=csv.reader(csvfile,delimiter=',')
#    for row in readCSV:
#        x1_test.extend([float(row[0])])
#        x2_test.extend([float(row[1])])

# Normalize test data
#for i in range(0,len(x1_test)):
#    x1_test[i]=(x1_test[i]-mean_x1)/dev_x1
#    x2_test[i]=(x2_test[i]-mean_x2)/dev_x2

#y_test=w1*x1_test+w2*x2_test+B
# print y_test


y_trainOut= w1*x1_data+w2*x2_data+B

#Revert training and testing inputs to old values
for i in range(0,len(x1_data)):
    x1_data[i]=x1_data[i]*dev_x1+mean_x1
    x2_data[i]=x2_data[i]*dev_x2+mean_x2

#for i in range(0,len(x1_test)):
#    x1_test[i]=x1_test[i]*dev_x1+mean_x1
#    x2_test[i]=x2_test[i]*dev_x2+mean_x2

#print Training output

X_trainOut = np.empty([0,2])
X_trainOut=np.vstack([X_trainOut,['----------------------------------------------------------------------','']])
X_trainOut=np.vstack([X_trainOut,['Actual value','Predicted Value']])
X_trainOut=np.vstack([X_trainOut,['----------------------------------------------------------------------','']])

for i in range(0, len(x1_data)):
   row_new = [y_data[i], y_trainOut[i]]
   X_trainOut=np.vstack([X_trainOut,row_new])

with open('TrOutput.csv','wb') as csvWriteFile:
    writeCSV=csv.writer(csvWriteFile,delimiter=",")
    writeCSV.writerows(X_trainOut)


#print Testing Output
#X = np.empty([0,1])
#for i in range(0, len(x1_test)):
#   row_new = [x1_test[i], x2_test[i], y_test[i]]
 #  row_new=[y_test[i]*100]
 #  X=np.vstack([X,row_new])

#with open('Output.csv','wb') as csvWriteFile:
#    writeCSV=csv.writer(csvWriteFile,delimiter=",")
#    writeCSV.writerows(X)

# print X

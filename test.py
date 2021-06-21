import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

tf.set_random_seed(7777)

input_col_counter = 18 
output_col_counter = 1 
 
len_sequence = 7
hidden_cell = 150  
hidden_cell2 = 20

forget_bias = 10  
keep_prob = 0.95
 
epoch_num =10400        
learning_rate = 0.02


def nomalization(x, w):

    weight = [-1, -1, 1, 1.1, 1.3, 1.7, 2.5]
    x[:,0] = [i * weight[2*int(j)] for i,j in zip(x[:,0], w[:,1])]
    x[:,1:] = [i * weight[2*int(j)] for i,j in zip(x[:,1:], w[:,0])]
  
    ary = np.asarray(x)  
    ary = ( x - x.min(0))/x.ptp(0)
    return (ary - ary.min(axis=0)) / (ary.max(axis=0) - ary.min(axis=0) + 1e-8) 
    
     
def nomalization2(x):
    ary = np.asarray(x)
    return (ary - ary.min()) / (ary.max() - ary.min()  + 1e-8)

def reverse_nomalization(org_x, x):
    org_ary = np.asarray(org_x)
    ary = np.asarray(x)
    return (ary * (org_ary.max(axis=0) - org_ary.min(axis=0) + 1e-8)) + org_ary.min(axis=0)
 
def reverse_nomalization2(org_x, x):
    org_ary = np.asarray(org_x)
    ary = np.asarray(x)
    return (ary * (org_ary.max() - org_ary.min() + 1e-8)) + org_ary.min()


def lstm_cell():
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_cell, forget_bias=forget_bias, state_is_tuple=True, activation=tf.nn.softsign)

    if keep_prob < 1.0:
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)

    cell2 = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_cell2, forget_bias=forget_bias, state_is_tuple=True, activation=tf.nn.softsign)
    
    
    return [cell, cell2]



names = ['구분별', '서울', '부산', '대구', '인천', '광주', '대전', '울산', '세종', '경기', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주', '검역','전국거리두기', '수도권거리두기','전체']
raw_dataframe = pd.read_csv('data.csv', names=names, encoding='utf-8' ) 
raw_dataframe.info() 

del raw_dataframe['구분별'] 
del raw_dataframe['검역'] 

covid_inform = raw_dataframe.values[1:].astype(np.float) 

area = covid_inform[:,:-3]
weight = covid_inform[:,-3:-1]
norm_area = nomalization(area, weight) 

infection = covid_inform[:,-1:]
norm_infection = nomalization2(infection) 

x = np.concatenate((norm_area, norm_infection), axis=1)
y = x[:, [-1]] 

dataX = [] 
dataY = [] 
 
for i in range(0, len(y) - len_sequence):
    _x = x[i : i+len_sequence]
    _y = y[i + len_sequence] 
    
    dataX.append(_x) 
    dataY.append(_y)
 

train_size = int(len(dataY) * 0.82)

test_size = len(dataY) - train_size

x_train = np.array(dataX[0:train_size])
y_train = np.array(dataY[0:train_size])
 
testX = np.array(dataX[train_size:len(dataX)])
testY = np.array(dataY[train_size:len(dataY)])
 
X = tf.placeholder(tf.float32, [None, len_sequence, input_col_counter])
Y = tf.placeholder(tf.float32, [None, 1])

targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])

cell_m = tf.contrib.rnn.MultiRNNCell(lstm_cell(), state_is_tuple=True)

hypo, _states = tf.nn.dynamic_rnn(cell_m, X, dtype=tf.float32)
hypo = tf.contrib.layers.fully_connected(hypo[:, -1], output_col_counter, activation_fn=tf.identity)
loss = tf.reduce_sum(tf.square(hypo - Y))
optimizer = tf.train.AdamOptimizer(learning_rate)

train = optimizer.minimize(loss)
rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(targets, predictions)))
 
test_predict = ''        
 
train_error_save = []
test_error_save = []

sess = tf.Session()
sess.run(tf.global_variables_initializer())
 
for epoch in range(epoch_num):
    _, _loss = sess.run([train, loss], feed_dict={X: x_train, Y: y_train})

    if ((epoch+1) % 50 == 0) or (epoch == epoch_num-1): 
        
        train_predict = sess.run(hypo, feed_dict={X: x_train})
        train_error = sess.run(rmse, feed_dict={targets: y_train, predictions: train_predict})
        
        test_predict = sess.run(hypo, feed_dict={X: testX})
        test_error = sess.run(rmse, feed_dict={targets: testY, predictions: test_predict})
        
        train_error_save.append(train_error)
        test_error_save.append(test_error)
        
        print("epoch: {}, train_error(A): {:4f}, test_error(B): {:4f}".format(epoch+1, train_error, test_error))
        

value_recent = np.array([x[len(x)-len_sequence : ]])

test_predict2 = sess.run(hypo, feed_dict={X: value_recent})
 
print("test_predict", test_predict2[0])

test_predict = reverse_nomalization2(infection,test_predict)
testY = reverse_nomalization2(infection,testY)

test_predict2 = reverse_nomalization2(infection,test_predict2)

for i, j in zip(testY, test_predict):
    print(i, j)

print("Tomorrow's COVID-19", int(test_predict2[0])) 



plt.figure(1)
plt.plot(train_error_save, 'r')
plt.plot(test_error_save, 'b')
plt.xlabel('epoch')
plt.ylabel('rmse')
plt.show()

plt.figure(2)
plt.plot(testY, 'r')
plt.plot(test_predict, 'b')
plt.xlabel('Time Period')
plt.ylabel('infection')
plt.show()
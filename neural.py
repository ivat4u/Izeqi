# -*- coding: utf-8 -*-
# plotRatio  容积率
# transactionDate 土地成交日期
#openingDate 开盘日期
#floorPrice   地均价
#openingPrice 开盘价格
# Price_1 Price_3 Price_6 开盘1、3、6月价格

import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
import pandas as pd
import  os

path = os.getcwd() + '\data\data\data_all4.csv'
data = pd.read_csv(path, header=None, names=['plotRatio',
                                             'transactionDate',
                                             'openingDate','floorPrice',
                                             'openingPrice','Price_1',
                                             'Price_3','Price_6'])

data

data2 = (data - data.mean()) / data.std()
data2.head()

# set y(target variable) and  X (training data)
cols = data2.shape[1]
X = data2.iloc[:,0:cols-4]
y = data2.iloc[:,cols-4:cols]
X.shape, y.shape

def forward_propagate(X, theta1, theta2):
    m = X.shape[0]

    a1 = np.insert(X, 0, values=np.ones(m), axis=1)
    z2 = a1 * theta1.T
    a2 = np.insert((z2), 0, values=np.ones(m), axis=1)
    z3 = a2 * theta2.T
    h = (z3)

    return a1, z2, a2, z3, h


def cost(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    # reshape the parameter array into parameter matrices for each layer
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    # run the feed-forward pass
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    # compute the cost
    J = 0
    for i in range(m):
        temp=np.power((h[i,:]-y[i,:]),2)
        J += np.sum(temp)
    J = J / 2*m

    return J

# initial setup
input_size = 4
hidden_size =17
num_labels=4
learning_rate = 0.5
punish_rate=1

# randomly initialize a parameter array of the size of the full network's parameters
params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.25

m = X.shape[0]
X = np.matrix(X)
y = np.matrix(y)

# unravel the parameter array into parameter matrices for each layer
theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

theta1.shape, theta2.shape
#((hideL, 4+1L), (4L, hide+1L))

a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
a1.shape, z2.shape, a2.shape, z3.shape, h.shape

#((mL, 4+1L), (mL, hideL), (mL, hide+1L), (mL, 4L), (mL, 4L))

cost(params, input_size, hidden_size, num_labels, X, y, learning_rate)
# The cost function, when the hypothetical matrix h is calculated,
# the total error between y and h is calculated using the cost equation.
def backprop(params, input_size, hidden_size, num_labels, X, y, learning_rate):

    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    # reshape the parameter array into parameter matrices for each layer
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    # run the feed-forward pass
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    # initializations
    J = 0
    delta1 = np.zeros(theta1.shape)  # (hide, 5)
    delta2 = np.zeros(theta2.shape)  # (1, hide+1)

    # compute the cost
    for i in range(m):
        temp=np.power((h[i,:]-y[i,:]),2)
        J += np.sum(temp)
    J = J / 2*m

    # add the cost regularization term
    J += (float(punish_rate) / (2 * m)) * (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2)))

    ##### end of cost function logic, below is the new part #####

    # perform backpropagation
    for t in range(m):
        a1t = a1[t,:]  # (1, 5)
        z2t = z2[t,:]  # (1, hide)
        a2t = a2[t,:]  # (1, hide+1)
        ht = h[t,:]  # (1, 4)
        yt = y[t,:]  # (1, 4)

        d3t = ht - yt  # (1, 4)

        z2t = np.insert(z2t, 0, values=np.ones(1))  # (1, hide+1)
        d2t = np.multiply((theta2.T * d3t.T).T, (z2t))  # (1, hide)

        delta1 = delta1 + (d2t[:,1:]).T * a1t
        delta2 = delta2 + d3t.T * a2t

    delta1 = delta1 / m
    delta2 = delta2 / m

    # add the gradient regularization term
    delta1[:,1:] = delta1[:,1:] + (theta1[:,1:] * punish_rate) / m
    delta2[:,1:] = delta2[:,1:] + (theta2[:,1:] * punish_rate) / m

    # unravel the gradient matrices into a single array
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))

    return J, grad

J, grad = backprop(params, input_size, hidden_size, num_labels, X, y, learning_rate)
J, grad.shape

from scipy.optimize import minimize

# minimize the objective function
fmin = minimize(fun=backprop, x0=params, args=(input_size, hidden_size, num_labels, X, y, learning_rate),
                method='TNC', jac=True, options={'maxiter': 10000})
fmin


X = np.matrix(X)
theta1 = np.matrix(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2 = np.matrix(np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
np.savetxt('new1.csv', theta1, delimiter=',')
np.savetxt('new2.csv', theta2, delimiter=',')

a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
h
#把每行放缩数据还原

def getpred(h):
   stdp = data.std()[0].ravel()
   meanp = data.mean()[0].ravel()
   stdt = data.std()[1].ravel()
   meant = data.mean()[1].ravel()
   stdo = data.std()[2].ravel()
   meano = data.mean()[2].ravel()
   stdf = data.std()[3].ravel()
   meanf = data.mean()[3].ravel()
   std0=data.std()[4].ravel()
   mean0=data.mean()[4].ravel()
   std1=data.std()[5].ravel()
   mean1=data.mean()[5].ravel()
   std3=data.std()[6].ravel()
   mean3=data.mean()[6].ravel()
   std6=data.std()[7].ravel()
   mean6=data.mean()[7].ravel()
   theta3 = np.matrix([stdp,meanp,stdt,meant,stdo,meano,stdf,meanf,std0, mean0, std1, mean1,std3,mean3,std6,mean6])
   np.savetxt('std_mean.csv', theta3, delimiter=',')

   #y_pred是我们预测的数据
   y_pred=h[:,0]*std0+mean0
   y_1pred=h[:,1]*std1+mean1
   y_3pred=h[:,2]*std3+mean3
   y_6pred=h[:,3]*std6+mean6
   e = np.concatenate((y_pred,y_1pred),axis=1)
   e=np.concatenate((e,y_3pred),axis=1)
   e=np.concatenate((e,y_6pred),axis=1)

   return e

e=getpred(h)



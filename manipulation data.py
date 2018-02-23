import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os

path = os.getcwd() + '/new1.csv'
theta1 = np.loadtxt(open(path,"rb"),delimiter=",",skiprows=0)
path = os.getcwd() + '/new2.csv'
theta2 = np.loadtxt(open(path,"rb"),delimiter=",",skiprows=0)
path = os.getcwd() + '/std_mean.csv'
theta3 = np.loadtxt(open(path,"rb"),delimiter=",",skiprows=0)
def getrequest(plotRatio,transactionDate,openingDate,floorPrice):

    return  plotRatio,transactionDate,openingDate,floorPrice

def forward_propagate(X, theta1, theta2):
    m = X.shape[0]

    a1 = np.insert(X, 0, values=np.ones(m), axis=1)
    z2 = a1 * theta1.T
    a2 = np.insert((z2), 0, values=np.ones(m), axis=1)
    z3 = a2 * theta2.T
    h = (z3)

    return a1, z2, a2, z3, h

def putresult(plotRatio,transactionDate,openingDate,floorPrice):
    stdp = theta3[0]
    meanp = theta3[1]
    stdt = theta3[2]
    meant = theta3[3]
    stdo = theta3[4]
    meano = theta3[5]
    stdf = theta3[6]
    meanf = theta3[7]
    std0 = theta3[8]
    mean0 = theta3[9]
    std1 = theta3[10]
    mean1 = theta3[11]
    std3 = theta3[12]
    mean3 = theta3[13]
    std6 = theta3[14]
    mean6 = theta3[15]
    p=(plotRatio-meanp)/stdp
    t = (transactionDate - meant) / stdt
    o = (openingDate - meano) / stdo
    f = (floorPrice - meanf) / stdf
    X=np.matrix([p,t,o,f])
    a1, z2, a2, z3, h =forward_propagate(X,theta1, theta2)
    y_pred = h[:, 0] * std0 + mean0
    y_1pred = h[:, 1] * std1 + mean1
    y_3pred = h[:, 2] * std3 + mean3
    y_6pred = h[:, 3] * std6 + mean6
    e = np.concatenate((y_pred, y_1pred), axis=1)
    e = np.concatenate((e, y_3pred), axis=1)
    e = np.concatenate((e, y_6pred), axis=1)
    openingPrice=e[0,0]
    Price_1=e[0,1]
    Price_3 = e[0, 2]
    Price_6 = e[0, 3]
    return openingPrice,Price_1,Price_3,Price_6

e,e1,e3,e6=putresult(1.5,2013.8,2016.8,6000)